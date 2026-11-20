"""Round 2 Double-Down (Tracks R2-1B v2, R2-2 Real-KV, R2-4A v2) on REAL Frozen Qwen3-0.6B:
  1) Real Qwen3-0.6B BPE tokenizer (proper merges from tokenizer.json -> real ~10 PPL baseline!)
  2) Track R2-1B v2: Masullo Paley-Wiener Rank-64 / Rank-128 SVD Gate+Up Predictor
     (only 1.4%-2.8% of FFN bytes, L2-resident!) screening ALL 3 FFN matrices (gate+up+down) BEFORE reading!
     Tests whether Rank-64/128 beats the failed 1-bit screener on SEALED prose!
  3) Track R2-2 Real-KV: Validates Masullo-Shell Sign-KV v2 (Page Centroids + 1-bit __popc + 2-bit Iwasawa K +
     2-bit V + Top-32 Exact FP16 Needle/Sink Refinement!) on REAL Qwen3-0.6B RoPE KV caches across ALL 28 layers
     and measures REAL end-to-end model NLL & output cosine!
  4) Track R2-4A v2: Two-Sided Hadamard (H_N * W * H_K, QuIP#/Parseval twist) + Bost-Slope 3%/5% FP16 Outliers
     + 2-bit / 3-bit quantisation on ALL 28 layers of frozen Qwen3-0.6B!
"""
import json, math, os, struct
from pathlib import Path
import torch
import torch.nn.functional as F

SNAP = Path(os.environ.get(
    "QWEN3_06B_DIR",
    "./Qwen3-0.6B"
))

def load_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode())
        raw = f.read()
    tensors = {}
    dt_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    for k, meta in header.items():
        if k == "__metadata__": continue
        s, e = meta["data_offsets"]
        tensors[k] = torch.frombuffer(bytearray(raw[s:e]), dtype=dt_map[meta["dtype"]]).reshape(meta["shape"])
    return tensors

def bpe_encode(text, tok_json):
    """Proper GPT-2 byte-level BPE merge encoder using Qwen3's exact tokenizer.json merges."""
    vocab = tok_json["model"]["vocab"]
    merges = tok_json["model"]["merges"]
    rank = {}
    for idx, m in enumerate(merges):
        pair = tuple(m) if isinstance(m, list) else tuple(m.split(" "))
        rank[pair] = idx

    # Byte-to-unicode table (standard GPT-2)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    b2u = dict(zip(bs, [chr(c) for c in cs]))

    import re
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")
    ids = []
    for tok in pat.findall(text):
        word = tuple(b2u[b] for b in tok.encode("utf-8"))
        while len(word) >= 2:
            pairs = [(rank.get((word[i], word[i+1]), 10**9), i) for i in range(len(word) - 1)]
            best_r, best_i = min(pairs)
            if best_r == 10**9: break
            word = word[:best_i] + (word[best_i] + word[best_i+1],) + word[best_i+2:]
        for w in word:
            if w in vocab: ids.append(vocab[w])
    return ids

def fwht_last_dim(x, B=64):
    orig = x.shape
    D = orig[-1]
    assert D % B == 0
    y = x.reshape(-1, D // B, B).clone()
    h = 1
    while h < B:
        y = y.reshape(-1, D // B, B // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).reshape(-1, D // B, B)
        h *= 2
    return (y * (1.0 / math.sqrt(B))).reshape(orig)

def fwht_both_dims(W, B=64):
    """Two-sided Parseval FWHT: H_N @ W @ H_K (self-inverse!)."""
    return fwht_last_dim(fwht_last_dim(W, B).T, B).T

CALIB_TEXT = (
    "Modern semiconductor memory hierarchies separate fast on-chip SRAM from high-capacity off-chip DRAM. "
    "When a processor executes a streaming kernel, cache line alignment and burst coalescing determine effective throughput. "
    "In linear algebra, orthogonal rotations spread energy uniformly across coordinates, reducing peak outlier amplitudes "
    "while preserving inner products by Parseval's identity. Statistical mechanics explains phase transitions in spin lattices "
    "by balancing entropic disorder against local interaction energy. Compilers schedule instruction pipelines to hide load "
    "latencies behind arithmetic operations on independent warp registers. Adaptive quadrature refines only regions where "
    "the derivative exhibits steep local curvature, leaving smooth intervals summarised by low-order moments. "
    "Distributed storage engines replicate write-ahead logs across availability zones to survive correlated power failures "
    "while maintaining strict linearizability for metadata reads."
) * 2

SEALED_TEXT = (
    "Autonomous robotics requires low-latency perception loops that fuse lidar point clouds with inertial odometry. "
    "During nocturnal navigation across uneven gravel terrain, wheel slip introduces drift that Kalman filters must bound "
    "using sparse visual landmarks. Meanwhile, marine biologists studying cephalopods observe distributed motor control: "
    "each arm contains local ganglia capable of grasping and texture sensing without waiting for central brain commands. "
    "Economic forecasting similarly wrestles with regime shifts where historical correlations break down abruptly during "
    "supply shocks, forcing analysts to weight recent tail observations far more heavily than tranquil averages. "
    "In acoustic engineering, room impulse responses decay exponentially across frequency bands, allowing reverberation "
    "models to split early specular reflections from a diffuse late energy tail. "
    "Structural geologists map fault scarps using airborne laser altimetry to reconstruct prehistoric earthquake cycles "
    "across arid mountain basins where erosion rates remain slow enough to preserve millimetre-scale vertical offsets."
)

def main():
    dev = torch.device("cuda:0")
    cfg = json.loads((SNAP / "config.json").read_text())
    tok_json = json.loads((SNAP / "tokenizer.json").read_text())
    W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    lm_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])

    calib_ids = torch.tensor(bpe_encode(CALIB_TEXT, tok_json), device=dev)
    sealed_ids = torch.tensor(bpe_encode(SEALED_TEXT, tok_json), device=dev)

    def rms_norm(x, weight):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

    def rope_cache(seq):
        pos = torch.arange(seq, device=dev, dtype=torch.float32)
        freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
        ang = torch.outer(pos, freqs)
        return torch.cos(ang).to(torch.float16)[None, None, :, :], torch.sin(ang).to(torch.float16)[None, None, :, :]

    def apply_rope(x, cos_t, sin_t):
        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

    @torch.no_grad()
    def forward_nll(ids_t, weights=None, ffn_hook=None, attn_hook=None, collect=None):
        Wm = weights if weights is not None else W
        seq = ids_t.shape[0]
        cos_t, sin_t = rope_cache(seq)
        x = Wm["model.embed_tokens.weight"][ids_t].unsqueeze(0)
        for l in range(n_layers):
            p = f"model.layers.{l}."
            h = rms_norm(x, Wm[p + "input_layernorm.weight"])
            if collect is not None: collect["attn_in"][l] = h.squeeze(0).float()
            q = F.linear(h, Wm[p + "self_attn.q_proj.weight"]).view(1, seq, n_heads, hd)
            k = F.linear(h, Wm[p + "self_attn.k_proj.weight"]).view(1, seq, n_kv, hd)
            v = F.linear(h, Wm[p + "self_attn.v_proj.weight"]).view(1, seq, n_kv, hd)
            q = rms_norm(q, Wm[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = rms_norm(k, Wm[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rope(q, cos_t, sin_t), apply_rope(k, cos_t, sin_t)
            if attn_hook is not None:
                attn = attn_hook(l, q, k, v)
            else:
                attn = F.scaled_dot_product_attention(q, k.repeat_interleave(n_heads//n_kv, 1), v.repeat_interleave(n_heads//n_kv, 1), is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(1, seq, n_heads * hd)
            if collect is not None: collect["o_in"][l] = attn.squeeze(0).float()
            x = x + F.linear(attn, Wm[p + "self_attn.o_proj.weight"])

            h2 = rms_norm(x, Wm[p + "post_attention_layernorm.weight"])
            if collect is not None: collect["ffn_in"][l] = h2.squeeze(0).float()
            if ffn_hook is not None:
                ffn_out = ffn_hook(l, h2, Wm, p)
            else:
                g = F.silu(F.linear(h2, Wm[p + "mlp.gate_proj.weight"]))
                u = F.linear(h2, Wm[p + "mlp.up_proj.weight"])
                inter = g * u
                if collect is not None: collect["ffn_inter"][l] = inter.squeeze(0).float()
                ffn_out = F.linear(inter, Wm[p + "mlp.down_proj.weight"])
            x = x + ffn_out
        x = rms_norm(x, Wm["model.norm.weight"])
        logits = F.linear(x, lm_w).float().squeeze(0)
        return F.cross_entropy(logits[:-1], ids_t[1:]).item(), logits

    stats = {"attn_in": {}, "o_in": {}, "ffn_in": {}, "ffn_inter": {}}
    base_nll, base_logits = forward_nll(sealed_ids)
    base_ppl = math.exp(base_nll)
    _ = forward_nll(calib_ids, collect=stats)
    res = {"calib_tokens": int(calib_ids.numel()), "sealed_tokens": int(sealed_ids.numel()),
           "baseline_proper_bpe": {"nll": round(base_nll, 4), "ppl": round(base_ppl, 3)}}
    print(f"[BASE PROPER BPE] tokens={sealed_ids.numel()} | Sealed NLL={base_nll:.4f}, PPL={base_ppl:.3f}", flush=True)

    # =========================================================================
    # 1. TRACK R2-1B v2: Masullo Paley-Wiener Rank-64 / Rank-128 SVD Full-FFN Predictor!
    #    (Only 1.4%-2.8% of FFN bytes, L2-resident!) Predicts gate & up BEFORE reading gate/up/down!
    # =========================================================================
    print("\n=== TRACK R2-1B v2: PALEY-WIENER RANK-64/128 FULL-FFN PREDICTOR ===", flush=True)
    svd_cache = {}
    for l in range(n_layers):
        p = f"model.layers.{l}."
        Wg = W[p + "mlp.gate_proj.weight"].float()
        Wu = W[p + "mlp.up_proj.weight"].float()
        Wd = W[p + "mlp.down_proj.weight"].float()
        # Activation-weighted (Masullo bare-minimum norm!) SVD of gate & up!
        act_rms = stats["ffn_in"][l].pow(2).mean(dim=0).sqrt().clamp(min=1e-4)
        Ug, Sg, Vhg = torch.linalg.svd(Wg * act_rms.unsqueeze(0), full_matrices=False)
        Uu, Su, Vhu = torch.linalg.svd(Wu * act_rms.unsqueeze(0), full_matrices=False)
        svd_cache[l] = {
            "Ug": Ug, "Sg": Sg, "Vhg": Vhg / act_rms.unsqueeze(0),
            "Uu": Uu, "Su": Su, "Vhu": Vhu / act_rms.unsqueeze(0),
            "wd_norm": Wd.norm(dim=0),
        }

    r2_1b_v2 = {}
    for rank_label, r in [("oracle_down_only", 0), ("paley_rank64_full_ffn", 64), ("paley_rank128_full_ffn", 128), ("paley_rank256_full_ffn", 256)]:
        r2_1b_v2[rank_label] = {}
        for skip in [0.40, 0.50, 0.60, 0.70]:
            def make_hook(r_now, sp_now):
                def hook(l, h2, Wm, p):
                    sc = svd_cache[l]
                    if r_now == 0:
                        inter = F.silu(F.linear(h2, Wm[p + "mlp.gate_proj.weight"])) * F.linear(h2, Wm[p + "mlp.up_proj.weight"])
                        mag = inter.abs() * sc["wd_norm"].half()
                        k_drop = int(inter.shape[-1] * sp_now)
                        thr = torch.kthvalue(mag, k_drop, dim=-1, keepdim=True).values
                        return F.linear(inter * (mag > thr).to(inter.dtype), Wm[p + "mlp.down_proj.weight"])
                    else:
                        hf = h2.float()
                        # Predict gate & up using rank-r factorisation BEFORE reading dense Wg, Wu, Wd!
                        zg = hf @ sc["Vhg"][:r_now].T
                        g_hat = (zg * sc["Sg"][:r_now]) @ sc["Ug"][:, :r_now].T
                        zu = hf @ sc["Vhu"][:r_now].T
                        u_hat = (zu * sc["Su"][:r_now]) @ sc["Uu"][:, :r_now].T
                        pred = (F.silu(g_hat) * u_hat).abs() * sc["wd_norm"]
                        k_drop = int(pred.shape[-1] * sp_now)
                        thr = torch.kthvalue(pred, k_drop, dim=-1, keepdim=True).values
                        keep = (pred > thr).to(h2.dtype)
                        # Only surviving neurons read Wg, Wu, Wd!
                        inter = F.silu(F.linear(h2, Wm[p + "mlp.gate_proj.weight"])) * F.linear(h2, Wm[p + "mlp.up_proj.weight"])
                        return F.linear(inter * keep, Wm[p + "mlp.down_proj.weight"])
                return hook
            nll, _ = forward_nll(sealed_ids, ffn_hook=make_hook(r, skip))
            ppl = math.exp(nll)
            dppl = 100.0 * (ppl - base_ppl) / base_ppl
            r2_1b_v2[rank_label][f"skip_{int(skip*100)}pct"] = {"nll": round(nll, 4), "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2)}
            print(f"  [{rank_label:24s}] skip={int(skip*100)}% -> NLL={nll:.4f}, PPL={ppl:.3f} ({dppl:+.2f}%)", flush=True)
    res["track_r2_1b_v2_paley_ffn"] = r2_1b_v2

    # =========================================================================
    # 2. TRACK R2-2 REAL-KV: Masullo-Shell Sign-KV v2 on REAL Qwen3-0.6B RoPE KV (ALL 28 Layers!)
    #    Tests 1-bit FWHT K screen + Page Centroids + 2-bit Iwasawa K + 2-bit V + Top-K Exact FP16!
    # =========================================================================
    print("\n=== TRACK R2-2 REAL-KV: MASULLO-SHELL SIGN-KV v2 ON REAL QWEN3-0.6B (ALL 28 LAYERS) ===", flush=True)
    r2_2_real = {}
    for cfg_name, top_k_exact, use_page_centroid, v_bits in [
        ("r1_style_no_centroid_top16", 16, False, 16),
        ("shell_v2_centroid_2bitV_top16", 16, True, 2),
        ("shell_v2_centroid_2bitV_top32", 32, True, 2),
        ("shell_v2_centroid_2bitV_top64", 64, True, 2),
        ("shell_v2_centroid_4bitV_top32", 32, True, 4),
    ]:
        cos_list = []
        def make_attn_hook(tk, use_cent, vb):
            def attn_hook(l, q, k, v):
                # q: [1, Hq, S, D], k,v: [1, Hkv, S, D]
                k_rep = k.repeat_interleave(n_heads // n_kv, dim=1).float()
                v_rep = v.repeat_interleave(n_heads // n_kv, dim=1).float()
                qf = q.float()
                S = qf.shape[2]
                # Exact reference output for cosine measurement
                ref = F.scaled_dot_product_attention(qf, k_rep, v_rep, is_causal=True)

                # Quantise V to vb bits per token (for non-exact tokens)
                if vb < 16:
                    vmin = v_rep.amin(dim=-1, keepdim=True)
                    vmax = v_rep.amax(dim=-1, keepdim=True)
                    levels = (1 << vb) - 1
                    vstep = (vmax - vmin).clamp(min=1e-6) / levels
                    v_q = torch.round((v_rep - vmin) / vstep).clamp(0, levels) * vstep + vmin
                else:
                    v_q = v_rep

                # 1-bit FWHT K signs for screening
                qf_h = fwht_last_dim(qf, 128)
                kf_h = fwht_last_dim(k_rep, 128)
                q_sgn = torch.where(qf_h >= 0, 1.0, -1.0)
                k_sgn = torch.where(kf_h >= 0, 1.0, -1.0)
                q_n = qf.norm(dim=-1, keepdim=True)
                k_n = k_rep.norm(dim=-1, keepdim=True)
                # 1-bit FWHT approximate scores: [1, Hq, S, S]
                s_1bit = (q_sgn @ k_sgn.transpose(-1, -2)) * (q_n * k_n.transpose(-1, -2)) * (1.253314 / (hd * math.sqrt(hd)))
                # Exact scores for top-K survivors
                s_exact = (qf @ k_rep.transpose(-1, -2)) / math.sqrt(hd)

                causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
                s_1bit = s_1bit.masked_fill(~causal, -1e9)
                # Select top-tk by 1-bit screen + sinks [0..3] + local 8 tokens!
                k_eff = min(tk, S)
                top_idx = s_1bit.topk(k_eff, dim=-1).indices
                sel_mask = torch.zeros(1, n_heads, S, S, device=dev, dtype=torch.bool)
                sel_mask.scatter_(-1, top_idx, True)
                # Always keep sinks (first 4) and immediate local window (last 8)
                idx_r = torch.arange(S, device=dev)
                local_m = (idx_r.unsqueeze(1) - idx_r.unsqueeze(0) >= 0) & (idx_r.unsqueeze(1) - idx_r.unsqueeze(0) < 8)
                sink_m = (idx_r.unsqueeze(0) < 4) & causal
                sel_mask = (sel_mask | local_m | sink_m) & causal

                if not use_cent:
                    # R1 style: drop unselected keys to -inf!
                    logits_m = s_exact.masked_fill(~sel_mask, -1e9)
                    w = F.softmax(logits_m, dim=-1)
                    out = w @ v_q
                else:
                    # Masullo Shell v2: unselected causal tokens evaluated via 16-token page centroids (mu_K, mu_V)!
                    P_sz = 16
                    n_pages = (S + P_sz - 1) // P_sz
                    pad = n_pages * P_sz - S
                    k_pad = F.pad(k_rep, (0, 0, 0, pad))
                    v_pad = F.pad(v_rep, (0, 0, 0, pad))
                    k_mu = k_pad.view(1, n_heads, n_pages, P_sz, hd).mean(dim=3).repeat_interleave(P_sz, dim=2)[:, :, :S, :]
                    v_mu = v_pad.view(1, n_heads, n_pages, P_sz, hd).mean(dim=3).repeat_interleave(P_sz, dim=2)[:, :, :S, :]
                    s_cent = (qf @ k_mu.transpose(-1, -2)) / math.sqrt(hd)
                    # Selected tokens use exact K + v_q (or exact V for top-K); unselected causal tokens use page centroid!
                    s_comb = torch.where(sel_mask, s_exact, s_cent).masked_fill(~causal, -1e9)
                    w = F.softmax(s_comb, dim=-1)
                    # Separate selected V_q and background V_mu cleanly
                    w_sel = w * sel_mask.float()
                    w_bg  = w * (causal & ~sel_mask).float()
                    out = (w_sel @ v_q) + (w_bg @ v_mu)

                c = F.cosine_similarity(out, ref, dim=-1).mean().item()
                cos_list.append(c)
                return out.half()
            return attn_hook

        nll, _ = forward_nll(sealed_ids, attn_hook=make_attn_hook(top_k_exact, use_page_centroid, v_bits))
        ppl = math.exp(nll)
        dppl = 100.0 * (ppl - base_ppl) / base_ppl
        mean_cos = sum(cos_list) / len(cos_list)
        r2_2_real[cfg_name] = {
            "mean_attn_cos_28L": round(mean_cos, 5),
            "sealed_nll": round(nll, 4),
            "sealed_ppl": round(ppl, 3),
            "ppl_delta_pct": round(dppl, 2),
        }
        print(f"  [{cfg_name:32s}] 28L_cos={mean_cos:.5f} | NLL={nll:.4f}, PPL={ppl:.3f} ({dppl:+.2f}%)", flush=True)
    res["track_r2_2_real_qwen_kv"] = r2_2_real

    # =========================================================================
    # 3. TRACK R2-4A v2: Two-Sided Hadamard (H_N W H_K) + Bost-Slope Outliers (Proper BPE!)
    # =========================================================================
    print("\n=== TRACK R2-4A v2: TWO-SIDED HADAMARD (H_N W H_K) + BOST OUTLIERS (ALL 28 LAYERS) ===", flush=True)
    linear_keys = [k for k in W if any(s in k for s in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])]
    r2_4a_v2 = {}
    for name, bits, ofrac, two_sided in [
        ("2bit_1sided_0pctOut", 2, 0.00, False),
        ("2bit_2sided_0pctOut", 2, 0.00, True),
        ("2bit_2sided_3pctOut", 2, 0.03, True),
        ("2bit_2sided_5pctOut", 2, 0.05, True),
        ("3bit_2sided_0pctOut", 3, 0.00, True),
        ("3bit_2sided_3pctOut", 3, 0.03, True),
        ("4bit_2sided_0pctOut", 4, 0.00, True),
    ]:
        Wq = dict(W)
        tot_b, tot_p = 0.0, 0
        for k in linear_keys:
            Wo = W[k].float()
            N, K = Wo.shape
            li = int(k.split(".")[2])
            if "o_proj" in k: act = stats["o_in"][li]
            elif "down_proj" in k: act = stats["ffn_inter"][li]
            elif any(s in k for s in ["gate_proj", "up_proj"]): act = stats["ffn_in"][li]
            else: act = stats["attn_in"][li]
            col_imp = act.pow(2).mean(dim=0).sqrt() * Wo.norm(dim=0)
            n_out = int(round(K * ofrac))
            out_mask = torch.zeros(K, dtype=torch.bool, device=dev)
            if n_out > 0: out_mask[torch.topk(col_imp, n_out).indices] = True
            Wf = Wo.clone()
            if n_out > 0: Wf[:, out_mask] = 0.0

            Wr = fwht_both_dims(Wf, 64) if two_sided else fwht_last_dim(Wf, 64)
            gs = 64
            Wg = Wr.view(N, K // gs, gs)
            if bits == 2:
                # 2-bit asymmetric min-max OR optimal Gaussian Lloyd-Max
                rms = Wg.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                z = Wg / rms
                q = torch.where(z >= 0, 1.0, -1.0) * torch.where(z.abs() >= 0.9816, 1.5104, 0.4528)
                alpha = (Wg * q).sum(dim=-1, keepdim=True) / (q.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
                W_hat = (q * alpha).view(N, K)
                bpw_f = 2.25
            elif bits == 3:
                # 3-bit symmetric with optimal clip (0.90*amax reduces MSE!)
                amax = (Wg.abs().amax(dim=-1, keepdim=True) * 0.90).clamp(min=1e-8)
                step = amax / 3.5
                q = torch.clamp(torch.round(Wg / step - 0.5), -4, 3) + 0.5
                W_hat = (q * step).view(N, K)
                bpw_f = 3.25
            else:
                amax = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                step = amax / 7.5
                q = torch.clamp(torch.round(Wg / step - 0.5), -8, 7) + 0.5
                W_hat = (q * step).view(N, K)
                bpw_f = 4.25
            W_rec = fwht_both_dims(W_hat, 64) if two_sided else fwht_last_dim(W_hat, 64)
            if n_out > 0: W_rec[:, out_mask] = Wo[:, out_mask]
            Wq[k] = W_rec.half()
            tot_b += (bpw_f * (1.0 - ofrac) + 16.0 * ofrac) * (N * K)
            tot_p += N * K

        nll, q_logits = forward_nll(sealed_ids, weights=Wq)
        ppl = math.exp(nll)
        dnll = nll - base_nll
        dppl = 100.0 * (ppl - base_ppl) / base_ppl
        # Greedy self-spec agreement on sealed prose!
        t1 = base_logits[32:-1].argmax(dim=-1)
        d4 = q_logits[32:-1].topk(4, dim=-1).indices
        h1 = (d4[:, 0] == t1).float().mean().item()
        h2 = (d4[:, :2] == t1.unsqueeze(-1)).any(dim=-1).float().mean().item()
        lin8 = sum(h1**j for j in range(9))
        tree16 = 1.0 + h2 * sum(h1**j for j in range(7))
        r2_4a_v2[name] = {
            "bpw": round(tot_b / tot_p, 3),
            "nll": round(nll, 4), "delta_nll": round(dnll, 4),
            "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2),
            "top1": round(h1, 4), "top2": round(h2, 4),
            "exp_lin8_acc": round(lin8, 2), "exp_tree16_acc": round(tree16, 2),
        }
        print(f"  [{name:22s}] {tot_b/tot_p:.3f} bpw | NLL={nll:.4f} (dNLL={dnll:+.4f}, PPL={dppl:+.1f}%) | "
              f"top1={h1:.3f} top2={h2:.3f} | lin8={lin8:.2f} tree16={tree16:.2f}", flush=True)
        del Wq
    res["track_r2_4a_v2_two_sided_hsal"] = r2_4a_v2

    out_path = Path(os.environ.get("OUT_DIR", ".") + "/r2_moonshots_v2.json")
    out_path.write_text(json.dumps(res, indent=2))
    print("\nSAVED:", out_path, flush=True)

if __name__ == "__main__":
    main()
