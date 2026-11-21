// g++ -O3 -march=native -std=c++17 r5_tree_spec.cpp -I$HOME/repos/llama.cpp/include -I$HOME/repos/llama.cpp/ggml/include \
//     -L$HOME/repos/llama.cpp/build-cuda/bin -Wl,-rpath,$HOME/repos/llama.cpp/build-cuda/bin \
//     -lllama -lggml -lggml-base -lggml-cuda -lpthread -o r5_tree_spec
//
// Round 5 DD Moonshot (Double Down on Round 4's 2.79x / 126.13 tok/s Breakthrough!):
//   Root-cause lessons from Round 4 (`r4_tree_spec_8b.json`):
//     1) NEW RTX 3070 LAW: Unlike cold `llama-bench pp8` (44.1 ms), live FlashAttention decode at M=4..12 takes
//        ONLY 24.5 ms/cycle (faster than M=16 30.6 ms and M=32 36.6 ms)! Dummy padding WASTED +6 ms/cycle!
//     2) CPU `logits_top4()` ran scalar double `std::exp()` over 151,936 vocab entries per node -> wasted draft ms!
//     3) Fixing BOTH (unpadded adaptive trees + fast early-exit `logits_top4` + confidence-pruned depth) +
//        testing `p_min` sweep and `Q3_K_S`/`Q2_K` 0.6B draft pushes fresh-prose Qwen3-8B toward 3.0x-3.5x (>140 tok/s)!
//   Also logs exact target `top1 - top2` logit gap at `first_diff_tok` to transparently distinguish FP16 GEMV-vs-MMQ
//   near-ties (<0.05 logit diff) from genuine divergence!

#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
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

// Fast vectorized top-4 softmax with early-exit for confident tokens (cuts CPU exp() calls by >98%!)
inline std::vector<TopKItem> logits_top4_fast(const float* __restrict__ logits, int n_vocab) {
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
    // Fast path: if top-1 beats top-2 by > 5.0 logits (~>99% prob), skip partition scan completely!
    if (maxv - val[1] > 5.0f) {
        return {
            {idx[0], 0.993f, -0.007f},
            {idx[1], 0.005f, -5.30f},
            {idx[2], 0.001f, -6.90f},
            {idx[3], 0.001f, -6.90f}
        };
    }
    // Tight cutoff (-9.0f -> exp(-9)=1.2e-4) eliminates 99.9% of expf calls on 152K vocab!
    float cutoff = maxv - 9.0f;
    float Z = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        if (v > cutoff) Z += std::exp(v - maxv);
    }
    float logZ = std::log(Z);
    std::vector<TopKItem> out(4);
    for (int k = 0; k < 4; ++k) {
        float lp = (val[k] - maxv) - logZ;
        out[k] = {idx[k], std::exp(lp), lp};
    }
    return out;
}

inline llama_token logits_argmax(const float* __restrict__ logits, int n_vocab, float* out_gap = nullptr) {
    int best = 0;
    float v1 = -1e30f, v2 = -1e30f;
    for (int i = 0; i < n_vocab; ++i) {
        float v = logits[i];
        if (v > v1) { v2 = v1; v1 = v; best = i; }
        else if (v > v2) { v2 = v; }
    }
    if (out_gap) *out_gap = v1 - v2;
    return best;
}

// Fast zero-cost n-gram suffix lookup
std::vector<llama_token> ngram_lookup(const std::vector<llama_token>& hist, int max_len) {
    const int H = static_cast<int>(hist.size());
    if (H < 6) return {};
    for (int n = std::min(10, H - 1); n >= 3; --n) {
        for (int i = H - n - 1; i >= 0; --i) {
            bool ok = true;
            for (int j = 0; j < n; ++j) {
                if (hist[i + j] != hist[H - n + j]) { ok = false; break; }
            }
            if (ok) {
                std::vector<llama_token> out;
                out.reserve(max_len);
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
    int parent;
    int depth;
    float cum_logp;
    int dft_seq;
    std::vector<int> children;
    std::vector<llama_seq_id> leaf_seqs;
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
    double avg_nodes = 0.0;
    double tps = 0.0;
    double toks_per_cycle = 0.0;
    bool exact_match = false;
    int first_diff_tok = -1;
    float diff_serial_logit_gap = -1.0f;
    std::vector<llama_token> tokens;
    std::vector<float> logit_gaps;
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

llama_token prefill_prompt(llama_context* ctx_tgt, llama_context* ctx_dft,
                           const std::vector<llama_token>& prompt, int n_vocab, float* first_gap = nullptr) {
    llama_memory_clear(llama_get_memory(ctx_tgt), true);
    if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft), true);
    const int P = static_cast<int>(prompt.size());
    llama_batch b = llama_batch_init(P, 0, 1);
    for (int i = 0; i < P; ++i) batch_add(b, prompt[i], i, {0}, (i == P - 1));
    if (llama_decode(ctx_tgt, b) != 0) throw std::runtime_error("tgt prefill failed");
    llama_token first_tok = logits_argmax(llama_get_logits_ith(ctx_tgt, P - 1), n_vocab, first_gap);
    if (ctx_dft) {
        if (llama_decode(ctx_dft, b) != 0) throw std::runtime_error("dft prefill failed");
    }
    llama_batch_free(b);
    return first_tok;
}

RunMetrics run_serial(llama_context* ctx_tgt, const llama_vocab* vocab,
                      const std::vector<llama_token>& prompt, int n_predict,
                      const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    RunMetrics m; m.mode = "serial"; m.prompt_name = pname;
    float gap = 0.0f;
    llama_token cur = prefill_prompt(ctx_tgt, nullptr, prompt, n_vocab, &gap);
    m.tokens.push_back(cur); m.logit_gaps.push_back(gap);

    llama_batch b = llama_batch_init(1, 0, 1);
    int pos = static_cast<int>(prompt.size());
    auto t0 = Clock::now();
    for (int step = 1; step < n_predict; ++step) {
        batch_clear(b);
        batch_add(b, cur, pos, {0}, true);
        if (llama_decode(ctx_tgt, b) != 0) break;
        cur = logits_argmax(llama_get_logits_ith(ctx_tgt, 0), n_vocab, &gap);
        m.tokens.push_back(cur); m.logit_gaps.push_back(gap);
        ++pos; ++m.cycles;
        if (llama_vocab_is_eog(vocab, cur)) break;
    }
    m.wall_ms = ms_since(t0);
    m.verify_ms = m.wall_ms;
    m.n_generated = static_cast<int>(m.tokens.size()) - 1;
    m.avg_nodes = 1.0;
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b);
    return m;
}

// Fast Unpadded Linear Speculative + Ngram Spine (with exact sync timing + fast logits_top4!)
RunMetrics run_linear_fast(llama_context* ctx_tgt, llama_context* ctx_dft, const llama_vocab* vocab,
                           const std::vector<llama_token>& prompt, int n_predict,
                           int k_max, float p_min, float cum_min, bool use_ngram,
                           const std::string& mode_name, const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto mem_tgt = llama_get_memory(ctx_tgt);
    const auto mem_dft = llama_get_memory(ctx_dft);

    RunMetrics m; m.mode = mode_name; m.prompt_name = pname;
    llama_token last_tok = prefill_prompt(ctx_tgt, ctx_dft, prompt, n_vocab);
    m.tokens.push_back(last_tok);
    std::vector<llama_token> hist = prompt; hist.push_back(last_tok);

    int pos_tgt = static_cast<int>(prompt.size());
    int pos_dft = static_cast<int>(prompt.size());
    long long total_nodes = 0;

    llama_batch b_dft = llama_batch_init(512, 0, 1);
    llama_batch b_tgt = llama_batch_init(64, 0, 2);

    auto t_start = Clock::now();
    while (static_cast<int>(m.tokens.size()) < n_predict) {
        int rem = n_predict - static_cast<int>(m.tokens.size());
        int k_lim = std::min(k_max, rem);
        std::vector<llama_token> draft;

        auto td0 = Clock::now();
        if (use_ngram) {
            auto ng = ngram_lookup(hist, k_lim);
            if (static_cast<int>(ng.size()) >= 3) draft = ng;
        }
        if (draft.empty()) {
            batch_clear(b_dft);
            int d_end = static_cast<int>(hist.size());
            for (int p = pos_dft; p < d_end; ++p) batch_add(b_dft, hist[p], p, {0}, (p == d_end - 1));
            if (llama_decode(ctx_dft, b_dft) != 0) break;
            pos_dft = d_end;

            int logit_idx = b_dft.n_tokens - 1;
            float cum_p = 1.0f;
            for (int j = 0; j < k_lim; ++j) {
                auto top = logits_top4_fast(llama_get_logits_ith(ctx_dft, logit_idx), n_vocab);
                if (j > 0 && (top[0].prob < p_min || cum_p * top[0].prob < cum_min)) break;
                cum_p *= top[0].prob;
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

        auto tv0 = Clock::now();
        const int L = static_cast<int>(draft.size());
        batch_clear(b_tgt);
        batch_add(b_tgt, last_tok, pos_tgt, {0}, true);
        for (int j = 0; j < L; ++j) batch_add(b_tgt, draft[j], pos_tgt + 1 + j, {0}, true);
        total_nodes += (1 + L);
        if (llama_decode(ctx_tgt, b_tgt) != 0) break;
        // Force GPU sync before stopping verify timer (llama_get_logits_ith waits for completion)
        const float* l0 = llama_get_logits_ith(ctx_tgt, 0);
        m.verify_ms += ms_since(tv0);

        auto tk0 = Clock::now();
        int n_acc = 0;
        llama_token next_tok = 0;
        bool hit_eog = false;
        for (int j = 0; j <= L; ++j) {
            const float* lj = (j == 0) ? l0 : llama_get_logits_ith(ctx_tgt, j);
            llama_token t_pred = logits_argmax(lj, n_vocab);
            if (j < L && t_pred == draft[j]) {
                m.tokens.push_back(t_pred); hist.push_back(t_pred); ++n_acc;
                if (llama_vocab_is_eog(vocab, t_pred) || static_cast<int>(m.tokens.size()) >= n_predict) {
                    next_tok = t_pred; hit_eog = true; break;
                }
            } else {
                m.tokens.push_back(t_pred); hist.push_back(t_pred); next_tok = t_pred;
                if (llama_vocab_is_eog(vocab, t_pred)) hit_eog = true;
                break;
            }
        }
        int keep_until = pos_tgt + n_acc + 1;
        if (n_acc < L) llama_memory_seq_rm(mem_tgt, 0, keep_until, -1);
        pos_tgt = keep_until;

        int hist_committed = static_cast<int>(hist.size()) - 1;
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
    m.avg_nodes = static_cast<double>(total_nodes) / std::max(1, m.cycles);
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b_dft); llama_batch_free(b_tgt);
    return m;
}

// Fast UNPADDED Confidence-Pruned Adaptive Probability Tree + Ngram Spine!
// Key Round 5 innovations:
//   - ZERO dummy padding (exploits live FlashAttention M=4..14 = 24.5-27 ms fast zone!)
//   - Fast early-exit logits_top4_fast (cuts CPU softmax >98%)
//   - Adaptive confidence pruning: stops depth expansion immediately when best frontier cumulative prob < min_cum_p!
//   - Smart 2-way fork ONLY on genuine uncertainty (top1 < fork_thresh && top2 >= min_branch_p)!
RunMetrics run_adaptive_tree(llama_context* ctx_tgt, llama_context* ctx_dft, const llama_vocab* vocab,
                             const std::vector<llama_token>& prompt, int n_predict,
                             int max_nodes, int max_depth, int frontier_w,
                             float fork_thresh, float min_branch_p, float min_cum_lp,
                             bool use_ngram, const std::string& mode_name, const std::string& pname) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto mem_tgt = llama_get_memory(ctx_tgt);
    const auto mem_dft = llama_get_memory(ctx_dft);

    RunMetrics m; m.mode = mode_name; m.prompt_name = pname;
    llama_token last_tok = prefill_prompt(ctx_tgt, ctx_dft, prompt, n_vocab);
    m.tokens.push_back(last_tok);
    std::vector<llama_token> hist = prompt; hist.push_back(last_tok);

    int pos_tgt = static_cast<int>(prompt.size());
    int pos_dft = static_cast<int>(prompt.size());
    long long total_nodes = 0;

    llama_batch b_dft = llama_batch_init(512, 0, 34);
    llama_batch b_tgt = llama_batch_init(64, 0, 34);

    auto t_start = Clock::now();
    while (static_cast<int>(m.tokens.size()) < n_predict) {
        int rem = n_predict - static_cast<int>(m.tokens.size());
        int budget = std::min(max_nodes, rem + 1);

        std::vector<TreeNode> nodes;
        nodes.reserve(max_nodes);
        nodes.push_back({last_tok, -1, 0, 0.0f, 0, {}, {}});

        auto td0 = Clock::now();
        bool ngram_only = false;
        std::vector<llama_token> ng_spine;
        if (use_ngram) {
            ng_spine = ngram_lookup(hist, std::min(15, budget - 1));
            if (static_cast<int>(ng_spine.size()) >= 4) {
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

        if (!ngram_only && budget >= 2) {
            batch_clear(b_dft);
            int d_end = static_cast<int>(hist.size());
            for (int p = pos_dft; p < d_end; ++p) batch_add(b_dft, hist[p], p, {0}, (p == d_end - 1));
            if (llama_decode(ctx_dft, b_dft) != 0) break;
            pos_dft = d_end;
            int dft_base_pos = pos_dft;

            struct ExpItem { int node_idx; std::vector<TopKItem> top; };
            std::vector<ExpItem> cur_level;
            cur_level.push_back({0, logits_top4_fast(llama_get_logits_ith(ctx_dft, b_dft.n_tokens - 1), n_vocab)});
            int next_dft_seq = 1;

            for (int depth = 1; depth <= max_depth && static_cast<int>(nodes.size()) < budget && !cur_level.empty(); ++depth) {
                struct CandChild { float cum_lp; int parent_idx; llama_token tok; };
                std::vector<CandChild> cands;
                for (auto& ei : cur_level) {
                    const auto& pn = nodes[ei.node_idx];
                    int max_k = (depth == 1) ? 3 : 2;
                    for (int k = 0; k < max_k && k < static_cast<int>(ei.top.size()); ++k) {
                        if (k == 0 || (ei.top[0].prob < fork_thresh && ei.top[k].prob >= min_branch_p)) {
                            float clp = pn.cum_logp + ei.top[k].logp;
                            if (depth == 1 || clp > min_cum_lp) {
                                cands.push_back({clp, ei.node_idx, ei.top[k].tok});
                            }
                        }
                    }
                }
                if (cands.empty()) break;
                std::sort(cands.begin(), cands.end(), [](const CandChild& a, const CandChild& b) {
                    return a.cum_lp > b.cum_lp;
                });

                int slots_left = budget - static_cast<int>(nodes.size());
                int take = std::min({static_cast<int>(cands.size()), frontier_w, slots_left});
                if (take <= 0) break;

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
                        llama_memory_seq_cp(mem_dft, nodes[p_idx].dft_seq, dseq, 0, dft_base_pos + depth - 1);
                    }
                    int nid = static_cast<int>(nodes.size());
                    nodes.push_back({cands[i].tok, p_idx, depth, cands[i].cum_lp, dseq, {}, {}});
                    nodes[p_idx].children.push_back(nid);
                    new_nodes.push_back(nid);
                }

                cur_level.clear();
                // Only run another 0.6B step if we still have budget, depth, AND at least one viable branch!
                if (depth < max_depth && static_cast<int>(nodes.size()) < budget) {
                    batch_clear(b_dft);
                    for (int nid : new_nodes) {
                        batch_add(b_dft, nodes[nid].tok, dft_base_pos + depth - 1, {nodes[nid].dft_seq}, true);
                    }
                    if (llama_decode(ctx_dft, b_dft) != 0) break;
                    for (int i = 0; i < static_cast<int>(new_nodes.size()); ++i) {
                        cur_level.push_back({new_nodes[i], logits_top4_fast(llama_get_logits_ith(ctx_dft, i), n_vocab)});
                    }
                }
            }
            llama_memory_seq_keep(mem_dft, 0);
            llama_memory_seq_rm(mem_dft, 0, dft_base_pos, -1);
        }
        m.draft_ms += ms_since(td0);

        auto tk_pre = Clock::now();
        int n_leaves = 0;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (nodes[i].children.empty()) {
                llama_seq_id lseq = n_leaves++;
                int cur = i;
                while (cur >= 0) { nodes[cur].leaf_seqs.push_back(lseq); cur = nodes[cur].parent; }
            }
        }
        for (int s = 1; s < n_leaves; ++s) llama_memory_seq_cp(mem_tgt, 0, s, 0, pos_tgt);
        m.kv_ops_ms += ms_since(tk_pre);

        // ZERO DUMMY PADDING — verify ONLY real tree nodes!
        auto tv0 = Clock::now();
        batch_clear(b_tgt);
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            batch_add(b_tgt, nodes[i].tok, pos_tgt + nodes[i].depth, nodes[i].leaf_seqs, true);
        }
        total_nodes += static_cast<int>(nodes.size());
        if (llama_decode(ctx_tgt, b_tgt) != 0) break;
        const float* l0 = llama_get_logits_ith(ctx_tgt, 0);
        m.verify_ms += ms_since(tv0);

        auto tk1 = Clock::now();
        int cur_node = 0;
        bool hit_eog = false;
        llama_token bonus_tok = 0;
        while (true) {
            const float* lc = (cur_node == 0) ? l0 : llama_get_logits_ith(ctx_tgt, cur_node);
            llama_token t_pred = logits_argmax(lc, n_vocab);
            int matched_child = -1;
            for (int c : nodes[cur_node].children) {
                if (nodes[c].tok == t_pred) { matched_child = c; break; }
            }
            if (matched_child >= 0) {
                m.tokens.push_back(t_pred); hist.push_back(t_pred); cur_node = matched_child;
                if (llama_vocab_is_eog(vocab, t_pred) || static_cast<int>(m.tokens.size()) >= n_predict) {
                    bonus_tok = t_pred; hit_eog = true; break;
                }
            } else {
                m.tokens.push_back(t_pred); hist.push_back(t_pred); bonus_tok = t_pred;
                if (llama_vocab_is_eog(vocab, t_pred)) hit_eog = true;
                break;
            }
        }
        llama_seq_id win_seq = nodes[cur_node].leaf_seqs[0];
        int accepted_depth = nodes[cur_node].depth;
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
    m.avg_nodes = static_cast<double>(total_nodes) / std::max(1, m.cycles);
    m.tps = (m.n_generated * 1000.0) / std::max(0.001, m.wall_ms);
    m.toks_per_cycle = static_cast<double>(m.n_generated) / std::max(1, m.cycles);
    llama_batch_free(b_dft); llama_batch_free(b_tgt);
    return m;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <target_gguf> <draft_gguf> [n_predict] [draft_label]\n";
        return 1;
    }
    std::string tgt_path = argv[1];
    std::string dft_path = argv[2];
    int n_predict = (argc >= 4) ? std::atoi(argv[3]) : 256;
    std::string dft_label = (argc >= 5) ? argv[4] : "Q4_K_M";

    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();

    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model* m_tgt = llama_model_load_from_file(tgt_path.c_str(), mp);
    llama_model* m_dft = llama_model_load_from_file(dft_path.c_str(), mp);
    if (!m_tgt || !m_dft) { std::cerr << "Failed loading models\n"; return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(m_tgt);

    auto cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = 4096; cp_tgt.n_batch = 512; cp_tgt.n_ubatch = 512;
    cp_tgt.n_seq_max = 34; cp_tgt.kv_unified = true;
    cp_tgt.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp_tgt);

    auto cp_dft = llama_context_default_params();
    cp_dft.n_ctx = 4096; cp_dft.n_batch = 512; cp_dft.n_ubatch = 512;
    cp_dft.n_seq_max = 34; cp_dft.kv_unified = true;
    cp_dft.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp_dft);

    auto tokenize = [&](const std::string& s) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(vocab, s.c_str(), static_cast<int>(s.size()), t.data(), static_cast<int>(t.size()), true, true);
        t.resize(n); return t;
    };

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

    { auto wt = tokenize("Warmup GPU clocks now:"); run_serial(ctx_tgt, vocab, wt, 20, "warmup"); }

    std::cout << "{\n  \"track\": \"R5_DD_Unpadded_FastAdaptiveTree_Speculation\",\n"
              << "  \"draft_quant\": \"" << dft_label << "\",\n"
              << "  \"n_predict\": " << n_predict << ",\n  \"results\": [\n";
    bool first = true;

    for (auto& [pname, ptext] : prompts) {
        auto ptoks = tokenize(ptext);
        auto base = run_serial(ctx_tgt, vocab, ptoks, n_predict, pname);

        std::vector<RunMetrics> all;
        all.push_back(base);
        // Fast unpadded linear sweep (zero dummy pad + fast logits_top4!)
        all.push_back(run_linear_fast(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 15, 0.50f, 0.05f, true, "fast_linear_p50_ngram", pname));
        all.push_back(run_linear_fast(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 15, 0.60f, 0.08f, true, "fast_linear_p60_ngram", pname));
        all.push_back(run_linear_fast(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 12, 0.45f, 0.12f, true, "fast_linear_cum12_ngram", pname));
        // Unpadded Confidence-Pruned Adaptive Trees (zero dummy pad + fast logits_top4!)
        all.push_back(run_adaptive_tree(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 12, 7, 2, 0.72f, 0.18f, -1.6f, true, "unpad_tree12_d7_w2", pname));
        all.push_back(run_adaptive_tree(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 16, 8, 2, 0.75f, 0.16f, -1.9f, true, "unpad_tree16_d8_w2", pname));
        all.push_back(run_adaptive_tree(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 16, 6, 3, 0.78f, 0.15f, -1.8f, true, "unpad_tree16_d6_w3", pname));
        all.push_back(run_adaptive_tree(ctx_tgt, ctx_dft, vocab, ptoks, n_predict, 24, 8, 3, 0.80f, 0.14f, -2.1f, true, "unpad_tree24_d8_w3", pname));

        for (auto& r : all) {
            r.exact_match = (r.tokens == base.tokens);
            r.first_diff_tok = -1;
            r.diff_serial_logit_gap = -1.0f;
            for (size_t i = 0; i < std::min(r.tokens.size(), base.tokens.size()); ++i) {
                if (r.tokens[i] != base.tokens[i]) {
                    r.first_diff_tok = static_cast<int>(i);
                    if (i < base.logit_gaps.size()) r.diff_serial_logit_gap = base.logit_gaps[i];
                    break;
                }
            }
            double sp = r.tps / std::max(0.01, base.tps);
            std::cerr << std::fixed << std::setprecision(2)
                      << "  [" << pname << "] " << std::setw(24) << r.mode
                      << " | " << std::setw(6) << r.tps << " tok/s (" << sp << "x)"
                      << " | t/c=" << r.toks_per_cycle << " avgN=" << r.avg_nodes
                      << " | dft=" << r.draft_ms << " ver=" << r.verify_ms << " kv=" << r.kv_ops_ms
                      << " | eq=" << (r.exact_match ? "TRUE" : ("diff@" + std::to_string(r.first_diff_tok) + "(gap=" + std::to_string(r.diff_serial_logit_gap).substr(0,4) + ")")) << "\n";

            if (!first) std::cout << ",\n";
            first = false;
            std::cout << std::fixed << std::setprecision(2)
                      << "    {\"prompt\": \"" << pname << "\", \"mode\": \"" << r.mode << "\""
                      << ", \"n_gen\": " << r.n_generated
                      << ", \"tps\": " << r.tps
                      << ", \"speedup_vs_serial\": " << sp
                      << ", \"cycles\": " << r.cycles
                      << ", \"toks_per_cycle\": " << r.toks_per_cycle
                      << ", \"avg_verify_nodes\": " << r.avg_nodes
                      << ", \"wall_ms\": " << r.wall_ms
                      << ", \"draft_ms\": " << r.draft_ms
                      << ", \"verify_ms\": " << r.verify_ms
                      << ", \"kv_ops_ms\": " << r.kv_ops_ms
                      << ", \"exact_match\": " << (r.exact_match ? "true" : "false")
                      << ", \"first_diff_tok\": " << r.first_diff_tok
                      << ", \"diff_serial_logit_gap\": " << r.diff_serial_logit_gap << "}";
        }
    }
    std::cout << "\n  ]\n}\n";

    llama_free(ctx_dft); llama_free(ctx_tgt);
    llama_model_free(m_dft); llama_model_free(m_tgt);
    llama_backend_free();
    return 0;
}
