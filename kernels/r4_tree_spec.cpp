// g++ -O3 -std=c++17 r4_tree_spec.cpp -I$HOME/repos/llama.cpp/include -I$HOME/repos/llama.cpp/ggml/include \
//     -L$HOME/repos/llama.cpp/build-cuda/bin -Wl,-rpath,$HOME/repos/llama.cpp/build-cuda/bin \
//     -lllama -lggml -lggml-base -lggml-cuda -lpthread -o r4_tree_spec
//
// Round 4 #1 Council Moonshot (Unanimous: Neon, Muse, Sol 6.2, Astra, Kimi):
//   Custom C++ libllama.so Padded-pp16 / Probability-Tree Speculative Decoder + Zero-Cost Ngram Spine!
//   Attacks the exact two bottlenecks capping stock llama-server at 93.83 tok/s (2.03x) on Qwen3-8B fresh prose:
//     1) Stock linear p_min=0.5 frequently stops at 4-10 tokens -> falls into llama.cpp's 44.1 ms `pp8` valley
//        instead of 28.8 ms `pp16` or 34.4 ms `pp32`! Padding to 16 immediately avoids `pp8`!
//     2) Linear chains discard the entire tail on one miss; a 16-node or 32-node entropy-forked probability tree
//        (built by batched M=2..4 Qwen3-0.6B expansion + free ngram spine) verifies all 16-32 candidates in ONE
//        Tensor Core pass and commits the longest exact target-greedy path + bonus token!
//
// Benchmarks head-to-head on IDENTICAL sealed fresh prose, reasoning, and code prompts:
//   - serial (M=1 baseline)
//   - linear_k15_p50 (reproducing stock unpadded linear)
//   - padded16_linear_p50 (exact pp16 padding fix!)
//   - tree16_prob_ngram (16-node batched probability tree + ngram spine -> pp16!)
//   - tree32_prob_ngram (32-node batched probability tree + ngram spine -> pp32!)

#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct TopKItem {
    llama_token tok;
    float prob;
    float logp;
};

// Fast top-K softmax probabilities from logits array (top-4 is plenty for tree branching!)
std::vector<TopKItem> logits_top4(const float* logits, int n_vocab) {
    // 1. Find top-4 logits + max in one linear scan
    int idx[4] = {0, 0, 0, 0};
    float val[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        if (v > val[3]) {
            int pos = 3;
            while (pos > 0 && v > val[pos - 1]) {
                val[pos] = val[pos - 1];
                idx[pos] = idx[pos - 1];
                --pos;
            }
            val[pos] = v;
            idx[pos] = i;
        }
    }
    float maxv = val[0];
    // 2. Compute exact partition Z = sum exp(v - maxv) (only needed for true prob; fast vectorized loop)
    double Z = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        float d = logits[i] - maxv;
        if (d > -20.0f) Z += std::exp(static_cast<double>(d));
    }
    float logZ = static_cast<float>(std::log(Z));
    std::vector<TopKItem> out;
    out.reserve(4);
    for (int k = 0; k < 4; ++k) {
        float lp = (val[k] - maxv) - logZ;
        out.push_back({idx[k], std::exp(lp), lp});
    }
    return out;
}

llama_token logits_argmax(const float* logits, int n_vocab) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

// Fast zero-cost n-gram lookup over token history (matches longest suffix n in [4..12])
std::vector<llama_token> ngram_lookup(const std::vector<llama_token>& hist, int max_len) {
    const int H = static_cast<int>(hist.size());
    if (H < 6) return {};
    for (int n = std::min(10, H - 1); n >= 3; --n) {
        // Search backwards for most recent matching n-gram
        for (int i = H - n - 1; i >= 0; --i) {
            bool ok = true;
            for (int j = 0; j < n; ++j) {
                if (hist[i + j] != hist[H - n + j]) { ok = false; break; }
            }
            if (ok) {
                std::vector<llama_token> out;
                for (int k = 0; k < max_len && (i + n + k) < H; ++k) {
                    out.push_back(hist[i + n + k]);
                }
                return out;
            }
        }
    }
    return {};
}

struct TreeNode {
    llama_token tok;
    int parent;       // -1 for root (node 0 = already-verified last token!)
    int depth;        // 0 for root
    float cum_logp;
    int dft_seq;      // seq_id inside draft ctx during expansion
    std::vector<int> children;
    std::vector<llama_seq_id> leaf_seqs; // all leaf seq_ids in this node's subtree
};

struct RunMetrics {
    std::string mode;
    std::string prompt_name;
    int n_generated = 0;
    double wall_ms = 0.0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double kv_ops_ms = 0.0;
    int cycles = 0;
    double tps = 0.0;
    double toks_per_cycle = 0.0;
    bool exact_match = false;
    int first_diff_tok = -1;
    std::vector<llama_token> tokens;
};

void batch_clear(llama_batch& b) { b.n_tokens = 0; }

void batch_add(llama_batch& b, llama_token id, llama_pos pos,
               const std::vector<llama_seq_id>& seq_ids, bool logits) {
    int i = b.n_tokens;
    b.token[i] = id;
    b.pos[i] = pos;
    b.n_seq_id[i] = static_cast<int32_t>(seq_ids.size());
    for (size_t s = 0; s < seq_ids.size(); ++s) b.seq_id[i][s] = seq_ids[s];
    b.logits[i] = logits ? 1 : 0;
    b.n_tokens++;
}

// Prefill both target (and optionally draft) cleanly on seq 0, returning first argmax token from target
llama_token prefill_prompt(llama_context* ctx_tgt, llama_context* ctx_dft,
                           const std::vector<llama_token>& prompt, int n_vocab) {
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft), true);

    const int P = static_cast<int>(prompt.size());
    llama_batch b = llama_batch_init(P, 0, 1);
    for (int i = 0; i < P; ++i) batch_add(b, prompt[i], i, {0}, (i == P - 1));
    if (llama_decode(ctx_tgt, b) != 0) throw std::runtime_error("tgt prefill failed");
    llama_token first_tok = logits_argmax(llama_get_logits_ith(ctx_tgt, P - 1), n_vocab);

    if (ctx_dft) {
        if (llama_decode(ctx_dft, b) != 0) throw std::runtime_error("dft prefill failed");
    }
    llama_batch_free(b);
    return first_tok;
}

// 1. Pure Serial Baseline (M=1)
RunMetrics run_serial(llama_context* ctx_tgt, const llama_vocab* vocab,
                      const std::vector<llama_token>& prompt, int n_predict,
                      const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    RunMetrics m;
    m.mode = "serial"; m.prompt_name = pname;
    llama_token cur = prefill_prompt(ctx_tgt, nullptr, prompt, n_vocab);
    m.tokens.push_back(cur);

    llama_batch b = llama_batch_init(1, 0, 1);
    int pos = static_cast<int>(prompt.size());
    auto t0 = Clock::now();
    for (int step = 1; step < n_predict; ++step) {
        batch_clear(b);
        batch_add(b, cur, pos, {0}, true);
        if (llama_decode(ctx_tgt, b) != 0) break;
        cur = logits_argmax(llama_get_logits_ith(ctx_tgt, 0), n_vocab);
        m.tokens.push_back(cur);
        ++pos;
        ++m.cycles;
        if (llama_vocab_is_eog(vocab, cur)) break;
    }
    m.wall_ms = ms_since(t0);
    m.verify_ms = m.wall_ms;
    m.n_generated = static_cast<int>(m.tokens.size()) - 1;
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b);
    return m;
}

// 2. Linear Speculative (with optional pp16 padding + optional ngram spine!)
//    pad_to = 0 (stock unpadded -> hits pp8 cliff!), pad_to = 16 (always pp16!), pad_to = 32 (always pp32!)
RunMetrics run_linear_spec(llama_context* ctx_tgt, llama_context* ctx_dft, const llama_vocab* vocab,
                           const std::vector<llama_token>& prompt, int n_predict,
                           int k_max, float p_min, int pad_to, bool use_ngram,
                           const std::string& mode_name, const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto mem_tgt = llama_get_memory(ctx_tgt);
    const auto mem_dft = llama_get_memory(ctx_dft);

    RunMetrics m; m.mode = mode_name; m.prompt_name = pname;
    llama_token last_tok = prefill_prompt(ctx_tgt, ctx_dft, prompt, n_vocab);
    m.tokens.push_back(last_tok);
    std::vector<llama_token> hist = prompt;
    hist.push_back(last_tok);

    int pos_tgt = static_cast<int>(prompt.size()); // position where `last_tok` must be placed into tgt KV!
    int pos_dft = static_cast<int>(prompt.size()); // position where `last_tok` must be placed into dft KV!

    llama_batch b_dft = llama_batch_init(512, 0, 1);
    llama_batch b_tgt = llama_batch_init(64, 0, 2);

    auto t_start = Clock::now();
    while (static_cast<int>(m.tokens.size()) < n_predict) {
        int rem = n_predict - static_cast<int>(m.tokens.size());
        int k_lim = std::min(k_max, rem);
        std::vector<llama_token> draft;

        // A. Zero-cost n-gram lookup first if enabled
        auto td0 = Clock::now();
        if (use_ngram) {
            auto ng = ngram_lookup(hist, k_lim);
            if (static_cast<int>(ng.size()) >= 3) draft = ng;
        }

        // B. Otherwise draft with 0.6B GPU model (early-stop when top-1 prob < p_min)
        if (draft.empty()) {
            // Sync draft KV up to `last_tok` (positions pos_dft .. hist.size()-1)
            batch_clear(b_dft);
            int d_start = pos_dft;
            int d_end = static_cast<int>(hist.size());
            for (int p = d_start; p < d_end; ++p) {
                batch_add(b_dft, hist[p], p, {0}, (p == d_end - 1));
            }
            if (llama_decode(ctx_dft, b_dft) != 0) break;
            pos_dft = d_end;

            int logit_idx = b_dft.n_tokens - 1;
            for (int j = 0; j < k_lim; ++j) {
                auto top = logits_top4(llama_get_logits_ith(ctx_dft, logit_idx), n_vocab);
                if (j > 0 && top[0].prob < p_min) break;
                draft.push_back(top[0].tok);
                if (j + 1 < k_lim) {
                    batch_clear(b_dft);
                    batch_add(b_dft, top[0].tok, pos_dft, {0}, true);
                    if (llama_decode(ctx_dft, b_dft) != 0) break;
                    ++pos_dft;
                    logit_idx = 0;
                }
            }
        }
        m.draft_ms += ms_since(td0);

        // C. Verify [last_tok, draft[0], ..., draft[L-1]] on Qwen3-8B in ONE call!
        //    If pad_to > 0 and (1 + L) < pad_to, PAD dummy tokens on unused seq_id=1 so ggml-cuda hits pp16 (28.8 ms), NEVER pp8 (44.1 ms)!
        auto tv0 = Clock::now();
        const int L = static_cast<int>(draft.size());
        batch_clear(b_tgt);
        batch_add(b_tgt, last_tok, pos_tgt, {0}, true);
        for (int j = 0; j < L; ++j) {
            batch_add(b_tgt, draft[j], pos_tgt + 1 + j, {0}, true);
        }
        int real_n = 1 + L;
        int pad_n = (pad_to > real_n) ? (pad_to - real_n) : 0;
        for (int p = 0; p < pad_n; ++p) {
            // Dummy padding tokens on seq_id=1 at pos=0 (logits=false, removed immediately after!)
            batch_add(b_tgt, 0, p, {1}, false);
        }
        if (llama_decode(ctx_tgt, b_tgt) != 0) break;
        m.verify_ms += ms_since(tv0);

        // D. Exact target-greedy verification walk
        auto tk0 = Clock::now();
        int n_acc = 0;
        llama_token next_tok = 0;
        bool hit_eog = false;
        for (int j = 0; j <= L; ++j) {
            llama_token t_pred = logits_argmax(llama_get_logits_ith(ctx_tgt, j), n_vocab);
            if (j < L && t_pred == draft[j]) {
                m.tokens.push_back(t_pred);
                hist.push_back(t_pred);
                ++n_acc;
                if (llama_vocab_is_eog(vocab, t_pred) || static_cast<int>(m.tokens.size()) >= n_predict) {
                    next_tok = t_pred; hit_eog = true; break;
                }
            } else {
                // Bonus / correction token from target!
                m.tokens.push_back(t_pred);
                hist.push_back(t_pred);
                next_tok = t_pred;
                if (llama_vocab_is_eog(vocab, t_pred)) hit_eog = true;
                break;
            }
        }
        // Roll back rejected suffix on tgt (keep pos <= pos_tgt + n_acc) and drop pad seq 1
        int keep_until = pos_tgt + n_acc + 1; // pos_tgt had last_tok, pos_tgt+1..pos_tgt+n_acc had accepted draft tokens
        if (n_acc < L) {
            llama_memory_seq_rm(mem_tgt, 0, keep_until, -1);
        }
        if (pad_n > 0) {
            llama_memory_seq_rm(mem_tgt, 1, -1, -1);
        }
        pos_tgt = keep_until;

        // Roll back draft KV to match exact accepted prefix!
        int hist_committed = static_cast<int>(hist.size()) - 1; // last token `next_tok` is NOT yet in draft KV
        if (pos_dft > hist_committed) {
            llama_memory_seq_rm(mem_dft, 0, hist_committed, -1);
            pos_dft = hist_committed;
        }
        last_tok = next_tok;
        ++m.cycles;
        m.kv_ops_ms += ms_since(tk0);
        if (hit_eog) break;
    }

    m.wall_ms = ms_since(t_start);
    m.n_generated = static_cast<int>(m.tokens.size()) - 1;
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b_dft);
    llama_batch_free(b_tgt);
    return m;
}

// 3. TRUE PROBABILITY-TREE SPECULATION (16-Node -> pp16, 32-Node -> pp32) + ZERO-COST NGRAM SPINE!
//    Uses batched M=2..4 frontier expansion on Qwen3-0.6B (ONLY 5-6 draft calls to fill a 16-32 node tree!)
//    and ONE SINGLE shared-prefix multi-seq_id tree verification call on Qwen3-8B!
RunMetrics run_tree_spec(llama_context* ctx_tgt, llama_context* ctx_dft, const llama_vocab* vocab,
                         const std::vector<llama_token>& prompt, int n_predict,
                         int tree_budget, int max_depth, int frontier_w, float fork_thresh, bool use_ngram,
                         const std::string& mode_name, const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto mem_tgt = llama_get_memory(ctx_tgt);
    const auto mem_dft = llama_get_memory(ctx_dft);

    RunMetrics m; m.mode = mode_name; m.prompt_name = pname;
    llama_token last_tok = prefill_prompt(ctx_tgt, ctx_dft, prompt, n_vocab);
    m.tokens.push_back(last_tok);
    std::vector<llama_token> hist = prompt;
    hist.push_back(last_tok);

    int pos_tgt = static_cast<int>(prompt.size());
    int pos_dft = static_cast<int>(prompt.size());

    llama_batch b_dft = llama_batch_init(512, 0, 34);
    llama_batch b_tgt = llama_batch_init(64, 0, 34);

    auto t_start = Clock::now();
    while (static_cast<int>(m.tokens.size()) < n_predict) {
        int rem = n_predict - static_cast<int>(m.tokens.size());
        int budget = std::min(tree_budget, rem + 1); // includes root node 0 (`last_tok`)!

        std::vector<TreeNode> nodes;
        nodes.reserve(tree_budget);
        nodes.push_back({last_tok, -1, 0, 0.0f, 0, {}, {}});

        auto td0 = Clock::now();

        // A. Fast path: if zero-cost n-gram finds a long match (>= 6 tokens), build a pure n-gram spine + 0 draft time!
        bool ngram_only = false;
        std::vector<llama_token> ng_spine;
        if (use_ngram) {
            ng_spine = ngram_lookup(hist, budget - 1);
            if (static_cast<int>(ng_spine.size()) >= 8) {
                // Build linear spine inside tree (zero 0.6B GPU calls!)
                int cur_p = 0;
                for (llama_token tk : ng_spine) {
                    int nid = static_cast<int>(nodes.size());
                    nodes.push_back({tk, cur_p, nodes[cur_p].depth + 1, 0.0f, 0, {}, {}});
                    nodes[cur_p].children.push_back(nid);
                    cur_p = nid;
                }
                ngram_only = true;
            }
        }

        // B. Batched Depth-by-Depth Probability Tree Expansion on Qwen3-0.6B (ONLY ~5-7 calls to fill 16-32 nodes!)
        if (!ngram_only && budget >= 2) {
            // 1. Sync draft KV up to `last_tok` on seq 0
            batch_clear(b_dft);
            int d_start = pos_dft;
            int d_end = static_cast<int>(hist.size());
            for (int p = d_start; p < d_end; ++p) {
                batch_add(b_dft, hist[p], p, {0}, (p == d_end - 1));
            }
            if (llama_decode(ctx_dft, b_dft) != 0) break;
            pos_dft = d_end;
            int dft_base_pos = pos_dft; // position of depth-1 children in draft KV!

            // Priority frontier of expandable nodes: (cum_logp, node_idx, top4_candidates)
            struct ExpItem { int node_idx; std::vector<TopKItem> top; };
            std::vector<ExpItem> cur_level;
            cur_level.push_back({0, logits_top4(llama_get_logits_ith(ctx_dft, b_dft.n_tokens - 1), n_vocab)});
            int next_dft_seq = 1;

            for (int depth = 1; depth <= max_depth && static_cast<int>(nodes.size()) < budget && !cur_level.empty(); ++depth) {
                // Collect candidate children across current frontier, sorted by cumulative probability!
                struct CandChild { float cum_lp; float p_step; int parent_idx; llama_token tok; int rank_k; };
                std::vector<CandChild> cands;
                for (auto& ei : cur_level) {
                    const auto& pn = nodes[ei.node_idx];
                    // Always propose rank-0 child; propose rank-1 (and rank-2 at depth 1) if probability is meaningful!
                    int max_k = (depth == 1) ? 3 : 2;
                    for (int k = 0; k < max_k && k < static_cast<int>(ei.top.size()); ++k) {
                        if (k == 0 || (ei.top[0].prob < fork_thresh && ei.top[k].prob >= 0.12f)) {
                            float clp = pn.cum_logp + ei.top[k].logp;
                            if (depth == 1 || clp > -2.6f) { // prune dead branches (<7.4% cumulative)
                                cands.push_back({clp, ei.top[k].prob, ei.node_idx, ei.top[k].tok, k});
                            }
                        }
                    }
                }
                // Inject short ngram hint at depth 1 if present and not duplicate!
                if (depth == 1 && !ng_spine.empty()) {
                    bool dup = false;
                    for (auto& c : cands) if (c.parent_idx == 0 && c.tok == ng_spine[0]) dup = true;
                    if (!dup) cands.push_back({-0.25f, 0.78f, 0, ng_spine[0], 9});
                }
                if (cands.empty()) break;

                std::sort(cands.begin(), cands.end(), [](const CandChild& a, const CandChild& b) {
                    return a.cum_lp > b.cum_lp;
                });

                int slots_left = budget - static_cast<int>(nodes.size());
                int take = std::min({static_cast<int>(cands.size()), frontier_w, slots_left});
                if (take <= 0) break;

                // Create tree nodes & assign draft seq_ids (first child of a parent reuses parent's dft_seq; siblings fork!)
                std::vector<int> new_nodes;
                std::unordered_map<int, bool> parent_reused;
                for (int i = 0; i < take; ++i) {
                    int p_idx = cands[i].parent_idx;
                    int dseq;
                    if (!parent_reused[p_idx]) {
                        dseq = nodes[p_idx].dft_seq;
                        parent_reused[p_idx] = true;
                    } else {
                        dseq = next_dft_seq++;
                        // Copy parent's KV prefix in draft ctx (O(1) metadata copy!)
                        llama_memory_seq_cp(mem_dft, nodes[p_idx].dft_seq, dseq, 0, dft_base_pos + depth - 1);
                    }
                    int nid = static_cast<int>(nodes.size());
                    nodes.push_back({cands[i].tok, p_idx, depth, cands[i].cum_lp, dseq, {}, {}});
                    nodes[p_idx].children.push_back(nid);
                    new_nodes.push_back(nid);
                }

                // If we still have budget & depth left, run ONE batched M=take decode on Qwen3-0.6B!
                cur_level.clear();
                if (depth < max_depth && static_cast<int>(nodes.size()) < budget) {
                    batch_clear(b_dft);
                    for (int nid : new_nodes) {
                        batch_add(b_dft, nodes[nid].tok, dft_base_pos + depth - 1, {nodes[nid].dft_seq}, true);
                    }
                    if (llama_decode(ctx_dft, b_dft) != 0) break;
                    for (int i = 0; i < static_cast<int>(new_nodes.size()); ++i) {
                        cur_level.push_back({new_nodes[i], logits_top4(llama_get_logits_ith(ctx_dft, i), n_vocab)});
                    }
                }
            }

            // Clean up all draft speculative KV beyond `dft_base_pos` and extra draft seq_ids!
            llama_memory_seq_keep(mem_dft, 0);
            llama_memory_seq_rm(mem_dft, 0, dft_base_pos, -1);
        }
        m.draft_ms += ms_since(td0);

        // C. Assign leaf seq_ids bottom-up so every tree node carries all descendant leaf seq_ids!
        auto tk_pre = Clock::now();
        int n_leaves = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].children.empty()) {
                llama_seq_id lseq = n_leaves++;
                // Walk up to root tagging every ancestor with `lseq`
                int cur = i;
                while (cur >= 0) {
                    nodes[cur].leaf_seqs.push_back(lseq);
                    cur = nodes[cur].parent;
                }
            }
        }
        // Share target prefix KV (0..pos_tgt) across all leaf seq_ids (O(1) metadata copy in llama.cpp!)
        for (int s = 1; s < n_leaves; ++s) {
            llama_memory_seq_cp(mem_tgt, 0, s, 0, pos_tgt);
        }
        m.kv_ops_ms += ms_since(tk_pre);

        // D. Verify ENTIRE TREE in ONE padded `pp16` or `pp32` Tensor Core call on Qwen3-8B!
        auto tv0 = Clock::now();
        batch_clear(b_tgt);
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            batch_add(b_tgt, nodes[i].tok, pos_tgt + nodes[i].depth, nodes[i].leaf_seqs, true);
        }
        // Pad up to tree_budget (16 or 32) on unused seq_id = 33 so ggml-cuda NEVER hits the 44.1 ms pp8 valley!
        int pad_n = (tree_budget > static_cast<int>(nodes.size())) ? (tree_budget - static_cast<int>(nodes.size())) : 0;
        for (int p = 0; p < pad_n; ++p) {
            batch_add(b_tgt, 0, p, {33}, false);
        }
        if (llama_decode(ctx_tgt, b_tgt) != 0) break;
        m.verify_ms += ms_since(tv0);

        // E. Walk target-greedy winning path down the tree!
        auto tk1 = Clock::now();
        int cur_node = 0;
        bool hit_eog = false;
        llama_token bonus_tok = 0;
        while (true) {
            llama_token t_pred = logits_argmax(llama_get_logits_ith(ctx_tgt, cur_node), n_vocab);
            int matched_child = -1;
            for (int c : nodes[cur_node].children) {
                if (nodes[c].tok == t_pred) { matched_child = c; break; }
            }
            if (matched_child >= 0) {
                m.tokens.push_back(t_pred);
                hist.push_back(t_pred);
                cur_node = matched_child;
                if (llama_vocab_is_eog(vocab, t_pred) || static_cast<int>(m.tokens.size()) >= n_predict) {
                    bonus_tok = t_pred; hit_eog = true; break;
                }
            } else {
                // Commit free bonus / correction token from target!
                m.tokens.push_back(t_pred);
                hist.push_back(t_pred);
                bonus_tok = t_pred;
                if (llama_vocab_is_eog(vocab, t_pred)) hit_eog = true;
                break;
            }
        }

        // F. Keep winning branch KV in O(1), trim losing branches & pad seq 31, rename winning seq -> 0!
        llama_seq_id win_seq = nodes[cur_node].leaf_seqs[0];
        int accepted_depth = nodes[cur_node].depth; // 0 if only bonus token committed, else depth >= 1
        int new_pos_tgt = pos_tgt + accepted_depth + 1;
        llama_memory_seq_keep(mem_tgt, win_seq);
        llama_memory_seq_rm(mem_tgt, win_seq, new_pos_tgt, -1);
        if (win_seq != 0) {
            llama_memory_seq_cp(mem_tgt, win_seq, 0, 0, new_pos_tgt);
            llama_memory_seq_keep(mem_tgt, 0);
        }
        pos_tgt = new_pos_tgt;
        last_tok = bonus_tok;
        ++m.cycles;
        m.kv_ops_ms += ms_since(tk1);
        if (hit_eog) break;
    }

    m.wall_ms = ms_since(t_start);
    m.n_generated = static_cast<int>(m.tokens.size()) - 1;
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b_dft);
    llama_batch_free(b_tgt);
    return m;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <target_gguf> <draft_gguf> [n_predict]\n";
        return 1;
    }
    std::string tgt_path = argv[1];
    std::string dft_path = argv[2];
    int n_predict = (argc >= 4) ? std::atoi(argv[3]) : 256;

    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model* m_tgt = llama_model_load_from_file(tgt_path.c_str(), mp);
    llama_model* m_dft = llama_model_load_from_file(dft_path.c_str(), mp);
    if (!m_tgt || !m_dft) { std::cerr << "Failed loading models\n"; return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(m_tgt);

    auto cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = 4096;
    cp_tgt.n_batch = 512;
    cp_tgt.n_ubatch = 512;
    cp_tgt.n_seq_max = 34;
    cp_tgt.kv_unified = true;
    cp_tgt.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp_tgt);

    auto cp_dft = llama_context_default_params();
    cp_dft.n_ctx = 4096;
    cp_dft.n_batch = 512;
    cp_dft.n_ubatch = 512;
    cp_dft.n_seq_max = 34;
    cp_dft.kv_unified = true;
    cp_dft.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp_dft);

    auto tokenize = [&](const std::string& s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), static_cast<int>(s.size()), t.data(), static_cast<int>(t.size()), true, true);
        t.resize(n);
        return t;
    };

    // Exact same sealed prompts as Round 2 (`r2_llama_prose_moonshots.py`) for 100% apples-to-apples comparison!
    std::vector<std::pair<std::string, std::string>> prompts = {
        {"prose",
         "Explain in clear, concrete engineering detail how memory bandwidth, L2 cache residency, and warp occupancy "
         "interact when decoding a transformer language model on an 8 GB consumer GPU. Cover attention KV caches, "
         "SwiGLU feed-forward layers, and why batching multiple candidate tokens changes the roofline."},
        {"reason",
         "A data centre operator has 3 servers with different power caps (80 W, 250 W, 400 W) and wants to minimise "
         "electricity cost per million tokens while keeping latency under 25 milliseconds per token. Walk through "
         "the exact trade-off between batching, quantisation bit-width, and speculative decoding step by step."},
        {"code",
         "Here is a Python CUDA benchmark script. Refactor it to add FP8/INT4 quantization timing, keep all existing functions, comments, and field names intact, and return the complete updated script:\n\n"
         "```python\nimport time, torch, json\n\ndef bench_ms(fn, warmup=10, iters=50):\n    for _ in range(warmup): fn()\n    torch.cuda.synchronize()\n    s = torch.cuda.Event(enable_timing=True)\n    e = torch.cuda.Event(enable_timing=True)\n    s.record()\n    for _ in range(iters): fn()\n    e.record()\n    torch.cuda.synchronize()\n    return s.elapsed_time(e) / iters\n\ndef run_roofline(dev):\n    results = {}\n    for kb in [128, 512, 1536, 3072, 6144, 16384, 65536]:\n        n = (kb * 1024) // 4\n        a = torch.empty(n, dtype=torch.float32, device=dev)\n        b = torch.empty(n, dtype=torch.float32, device=dev)\n        ms = bench_ms(lambda: b.copy_(a), warmup=20, iters=60)\n        results[f\"{kb}KB\"] = round((2 * kb * 1024) / (ms * 1e-3) / 1e9, 1)\n    return results\n```\nReturn the complete refactored Python script now:\n```python\n"}
    };

    // Warmup clocks
    { auto wt = tokenize("Warmup GPU clocks now:"); run_serial(ctx_tgt, vocab, wt, 20, "warmup"); }

    std::cout << "{\n  \"track\": \"R4-3A_LibLlama_PaddedPP16_ProbabilityTree_Speculation\",\n"
              << "  \"n_predict\": " << n_predict << ",\n  \"results\": [\n";
    bool first = true;

    for (auto& [pname, ptext] : prompts) {
        auto ptoks = tokenize(ptext);
        auto base = run_serial(ctx_tgt, vocab, ptoks, n_predict, pname);

        std::vector<RunMetrics> all;
        all.push_back(base);
        // Unpadded linear (exposes pp8 cliff!)
        all.push_back(run_linear_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 15, 0.50f, 0, true, "linear_unpadded_p50_ngram", pname));
        // Padded to 16 (avoids pp8 cliff -> always pp16!)
        all.push_back(run_linear_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 15, 0.50f, 16, true, "linear_padded16_p50_ngram", pname));
        // Padded to 16 with lower p_min=0.30 (fills more of the free pp16 slots!)
        all.push_back(run_linear_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 15, 0.30f, 16, true, "linear_padded16_p30_ngram", pname));
        // 16-Node Entropy-Forked Probability Tree (depth=6, frontier_w=3, fork<0.85, pp16!)
        all.push_back(run_tree_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 16, 7, 3, 0.85f, true, "tree16_d7_w3_ngram", pname));
        // 16-Node Deeper Narrow Tree (depth=10, frontier_w=2, fork<0.75, pp16!)
        all.push_back(run_tree_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 16, 10, 2, 0.75f, true, "tree16_d10_w2_ngram", pname));
        // 32-Node Entropy-Forked Probability Tree (depth=10, frontier_w=4, fork<0.85, pp32!)
        all.push_back(run_tree_spec(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 32, 10, 4, 0.85f, true, "tree32_d10_w4_ngram", pname));

        for (auto& r : all) {
            r.exact_match = (r.tokens == base.tokens);
            r.first_diff_tok = -1;
            for (size_t i = 0; i < std::min(r.tokens.size(), base.tokens.size()); ++i) {
                if (r.tokens[i] != base.tokens[i]) { r.first_diff_tok = static_cast<int>(i); break; }
            }
            double sp = r.tps / std::max(0.01, base.tps);
            std::cerr << std::fixed << std::setprecision(2)
                      << "  [" << pname << "] " << std::setw(26) << r.mode
                      << " | " << std::setw(6) << r.tps << " tok/s (" << sp << "x)"
                      << " | tok/cyc=" << r.toks_per_cycle
                      << " | dft=" << r.draft_ms << "ms ver=" << r.verify_ms << "ms kv=" << r.kv_ops_ms << "ms"
                      << " | eq=" << (r.exact_match ? "TRUE" : ("diff@" + std::to_string(r.first_diff_tok))) << "\n";

            if (!first) std::cout << ",\n";
            first = false;
            std::cout << std::fixed << std::setprecision(2)
                      << "    {\"prompt\": \"" << pname << "\", \"mode\": \"" << r.mode << "\""
                      << ", \"n_gen\": " << r.n_generated
                      << ", \"tps\": " << r.tps
                      << ", \"speedup_vs_serial\": " << sp
                      << ", \"cycles\": " << r.cycles
                      << ", \"toks_per_cycle\": " << r.toks_per_cycle
                      << ", \"wall_ms\": " << r.wall_ms
                      << ", \"draft_ms\": " << r.draft_ms
                      << ", \"verify_ms\": " << r.verify_ms
                      << ", \"kv_ops_ms\": " << r.kv_ops_ms
                      << ", \"exact_match\": " << (r.exact_match ? "true" : "false")
                      << ", \"first_diff_tok\": " << r.first_diff_tok << "}";
        }
    }
    std::cout << "\n  ]\n}\n";

    llama_free(ctx_dft); llama_free(ctx_tgt);
    llama_model_free(m_dft); llama_model_free(m_tgt);
    llama_backend_free();
    return 0;
}
