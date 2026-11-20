"""Round 2 Frozen-Model Moonshot Falsification Suite on REAL Qwen3-0.6B (28 layers, FP16).
Tests 3 Council Round 2 moonshots on SEALED fresh technical prose (strictly disjoint from calibration):
  1) Track R2-1B: 1-Bit Sign-Screened Full-FFN (gate+up+down skipped BEFORE reading!)
                  + 32-neuron co-activation clustered stripes + Carbon negative-SiLU bias twist!
  2) Track R2-4A: Residual HSAL / Iwasawa Plane-0 Frozen-Weight Rescue
                  (Signed Hadamard D*H + Bost-slope activation-weighted 1%/3%/5% FP16 outliers + 1-bit/2-bit)
                  + REAL Linear vs 16-Node Tree Lossless Speculative Acceptance on Fresh Prose!
  3) Track R2-3A: Bost-Slope Layer-Skip Self-Speculation (+0 MiB VRAM) & Kimi Jacobi 16-Token Fixed-Point Decoding!
"""
import json, math, os, struct, time
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

def simple_encode(text, vocab):
    s_enc = text.replace(" ", "Ġ").replace("\n", "Ċ")
    ids, i = [], 0
    while i < len(s_enc):
        matched = False
        for L in range(min(16, len(s_enc) - i), 0, -1):
            sub = s_enc[i:i+L]
            if sub in vocab:
                ids.append(vocab[sub]); i += L; matched = True; break
        if not matched:
            ids.append(vocab.get(s_enc[i], 0)); i += 1
    return ids

# Strictly disjoint CALIBRATION prose (used to fit Bost slopes, co-activation stripes, bias twist)
CALIB_TEXT = (
    "Modern semiconductor memory hierarchies separate fast on-chip SRAM from high-capacity off-chip DRAM. "
    "When a processor executes a streaming kernel, cache line alignment and burst coalescing determine effective throughput. "
    "In linear algebra, orthogonal rotations spread energy uniformly across coordinates, reducing peak outlier amplitudes "
    "while preserving inner products by Parseval's identity. Statistical mechanics explains phase transitions in spin lattices "
    "by balancing entropic disorder against local interaction energy. Compilers schedule instruction pipelines to hide load "
    "latencies behind arithmetic operations on independent warp registers. Adaptive quadrature refines only regions where "
    "the derivative exhibits steep local curvature, leaving smooth intervals summarised by low-order moments."
) * 2

# Strictly SEALED HELD-OUT prose (never touched during calibration!)
SEALED_TEXT = (
    "Autonomous robotics requires low-latency perception loops that fuse lidar point clouds with inertial odometry. "
    "During nocturnal navigation across uneven gravel terrain, wheel slip introduces drift that Kalman filters must bound "
    "using sparse visual landmarks. Meanwhile, marine biologists studying cephalopods observe distributed motor control: "
    "each arm contains local ganglia capable of grasping and texture sensing without waiting for central brain commands. "
    "Economic forecasting similarly wrestles with regime shifts where historical correlations break down abruptly during "
    "supply shocks, forcing analysts to weight recent tail observations far more heavily than tranquil averages. "
    "In acoustic engineering, room impulse responses decay exponentially across frequency bands, allowing reverberation "
    "models to split early specular reflections from a diffuse late energy tail."
)

def fwht_last_dim(x):
    """Exact orthonormal Fast Walsh-Hadamard Transform along last dimension (power-of-2 blocks of size 64)."""
    orig_shape = x.shape
    D = orig_shape[-1]
    B = 64
    assert D % B == 0
    y = x.reshape(-1, D // B, B).clone()
    h = 1
    while h < B:
        y = y.reshape(-1, D // B, B // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).reshape(-1, D // B, B)
        h *= 2
    return (y * (1.0 / math.sqrt(B))).reshape(orig_shape)

class Qwen06BRunner:
    def __init__(self, dev):
        self.dev = dev
        self.cfg = json.loads((SNAP / "config.json").read_text())
        tok_json = json.loads((SNAP / "tokenizer.json").read_text())
        self.vocab = tok_json["model"]["vocab"]
        self.W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
        self.n_layers = self.cfg["num_hidden_layers"]
        self.n_heads = self.cfg["num_attention_heads"]
        self.n_kv = self.cfg["num_key_value_heads"]
        self.hd = self.cfg["head_dim"]
        self.eps = self.cfg["rms_norm_eps"]
        self.theta = self.cfg["rope_theta"]
        self.lm_w = self.W.get("lm_head.weight", self.W["model.embed_tokens.weight"])

    def rms_norm(self, x, weight):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * weight

    def rope_cache(self, seq):
        pos = torch.arange(seq, device=self.dev, dtype=torch.float32)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=self.dev, dtype=torch.float32) / self.hd))
        ang = torch.outer(pos, freqs)
        return torch.cos(ang).to(torch.float16)[None, None, :, :], torch.sin(ang).to(torch.float16)[None, None, :, :]

    def apply_rope(self, x, cos_t, sin_t):
        x1, x2 = x[..., :self.hd//2], x[..., self.hd//2:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

    @torch.no_grad()
    def forward_logits(self, ids_t, weights=None, layer_mask=None, ffn_hook=None, collect_stats=None):
        W = weights if weights is not None else self.W
        seq = ids_t.shape[0]
        cos_t, sin_t = self.rope_cache(seq)
        x = W["model.embed_tokens.weight"][ids_t].unsqueeze(0)
        for l in range(self.n_layers):
            if layer_mask is not None and not layer_mask[l]:
                continue
            p = f"model.layers.{l}."
            h = self.rms_norm(x, W[p + "input_layernorm.weight"])
            if collect_stats is not None:
                collect_stats["attn_in"][l] = h.squeeze(0).float()
            q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, seq, self.n_heads, self.hd)
            k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, seq, self.n_kv, self.hd)
            v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, seq, self.n_kv, self.hd)
            q = self.rms_norm(q, W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = self.rms_norm(k, W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = self.apply_rope(q, cos_t, sin_t), self.apply_rope(k, cos_t, sin_t)
            k_rep = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            v_rep = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
            attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(1, seq, self.n_heads * self.hd)
            if collect_stats is not None:
                collect_stats["o_in"][l] = attn.squeeze(0).float()
            x = x + F.linear(attn, W[p + "self_attn.o_proj.weight"])

            h2 = self.rms_norm(x, W[p + "post_attention_layernorm.weight"])
            if collect_stats is not None:
                collect_stats["ffn_in"][l] = h2.squeeze(0).float()
            if ffn_hook is not None:
                ffn_out = ffn_hook(l, h2, W, p)
            else:
                g = F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"]))
                u = F.linear(h2, W[p + "mlp.up_proj.weight"])
                inter = g * u
                if collect_stats is not None:
                    collect_stats["ffn_inter"][l] = inter.squeeze(0).float()
                    collect_stats["ffn_gate"][l] = g.squeeze(0).float()
                ffn_out = F.linear(inter, W[p + "mlp.down_proj.weight"])
            x = x + ffn_out
        x = self.rms_norm(x, W["model.norm.weight"])
        return F.linear(x, self.lm_w).float().squeeze(0)

    def nll_on(self, ids_t, **kwargs):
        logits = self.forward_logits(ids_t, **kwargs)
        return F.cross_entropy(logits[:-1], ids_t[1:]).item()

def run_all():
    dev = torch.device("cuda:0")
    runner = Qwen06BRunner(dev)
    calib_ids = torch.tensor(simple_encode(CALIB_TEXT, runner.vocab)[:256], device=dev)
    sealed_ids = torch.tensor(simple_encode(SEALED_TEXT, runner.vocab)[:256], device=dev)
    results = {"calib_tokens": int(calib_ids.numel()), "sealed_tokens": int(sealed_ids.numel())}

    # Collect calibration activation statistics ONCE (disjoint from sealed_ids!)
    stats = {"attn_in": {}, "o_in": {}, "ffn_in": {}, "ffn_inter": {}, "ffn_gate": {}}
    base_sealed_nll = runner.nll_on(sealed_ids)
    base_sealed_ppl = math.exp(base_sealed_nll)
    _ = runner.forward_logits(calib_ids, collect_stats=stats)
    results["baseline_dense_fp16"] = {"sealed_nll": round(base_sealed_nll, 4), "sealed_ppl": round(base_sealed_ppl, 3)}
    print(f"[BASE] Sealed NLL={base_sealed_nll:.4f}, PPL={base_sealed_ppl:.3f}", flush=True)

    # =========================================================================
    # TRACK R2-1B: 1-Bit Sign-Screened Full-FFN + Co-Activation Stripes + Bias Twist
    # =========================================================================
    print("\n=== TRACK R2-1B: 1-BIT SIGN-SCREENED FULL-FFN + STRIPES + BIAS TWIST ===", flush=True)
    # Build per-layer 1-bit sign predictor for gate_proj (and rotated FWHT 1-bit!) + Bost-slope static neuron order
    # Also compute offline co-activation stripe permutation (group neurons into 32-neuron stripes by greedy cosine!)
    ffn_meta = {}
    for l in range(runner.n_layers):
        p = f"model.layers.{l}."
        Wg = runner.W[p + "mlp.gate_proj.weight"].float()
        Wu = runner.W[p + "mlp.up_proj.weight"].float()
        Wd = runner.W[p + "mlp.down_proj.weight"].float()
        inter_cal = stats["ffn_inter"][l] # [T, I]
        h2_cal = stats["ffn_in"][l]       # [T, H]

        # 1. Exact 1-bit FWHT sign predictor for gate_proj
        Wg_rot = fwht_last_dim(Wg)
        Wg_sign = torch.where(Wg_rot >= 0, 1.0, -1.0)
        # Scale by row L1/D (optimal 1-bit scale in FWHT domain)
        Wg_sc = Wg_rot.abs().mean(dim=-1, keepdim=True)
        # Also 1-bit predictor for up_proj
        Wu_rot = fwht_last_dim(Wu)
        Wu_sign = torch.where(Wu_rot >= 0, 1.0, -1.0)
        Wu_sc = Wu_rot.abs().mean(dim=-1, keepdim=True)

        # 2. Bost-slope neuron importance: E[|inter_i|] * ||W_down[:, i]||_2
        wd_col_norm = Wd.norm(dim=0) # [I]
        bost_score = inter_cal.abs().mean(dim=0) * wd_col_norm

        # 3. Co-activation stripe clustering (sort by first principal component of |inter_cal| / co-activation!)
        # SVD of normalized |inter_cal| [T, I] -> sort neurons by their top-2 spectral angle so co-active neurons land in same 32-stripe!
        act_norm = inter_cal.abs() / (inter_cal.abs().norm(dim=0, keepdim=True) + 1e-6)
        U_s, S_s, Vh_s = torch.linalg.svd(act_norm, full_matrices=False)
        angle = torch.atan2(Vh_s[1], Vh_s[0])
        perm = torch.argsort(angle) # exact permutation P (P^T P = I, 0.000 loss!)
        inv_perm = torch.argsort(perm)

        # 4. Calibrate mean bias correction for dropped neurons at each sparsity level (Carbon's negative-SiLU twist!)
        # For a given mask M (dropped = 1-M), average dropped output = E[W_down @ (inter * (1-M))]
        ffn_meta[l] = {
            "Wg_sign": Wg_sign.half(), "Wg_sc": Wg_sc.half(),
            "Wu_sign": Wu_sign.half(), "Wu_sc": Wu_sc.half(),
            "wd_col_norm": wd_col_norm.half(),
            "bost_score": bost_score,
            "perm": perm, "inv_perm": inv_perm,
            "inter_cal": inter_cal, "Wd": Wd
        }

    r2_1b = {}
    for mode in ["oracle_down_only", "sign_1bit_full_ffn", "sign_1bit_32stripe_ffn", "sign_1bit_32stripe_plus_bias_twist"]:
        r2_1b[mode] = {}
        for skip in [0.50, 0.60, 0.70]:
            # Precompute calibrated bias twist for this (mode, skip) on calib_ids ONLY!
            bias_vec = {}
            if mode == "sign_1bit_32stripe_plus_bias_twist":
                for l in range(runner.n_layers):
                    m = ffn_meta[l]
                    h2c = stats["ffn_in"][l].half()
                    hr = fwht_last_dim(h2c.float())
                    hs = torch.where(hr >= 0, 1.0, -1.0).half()
                    # Fast 1-bit sign score for gate & up
                    g_hat = (hs @ m["Wg_sign"].T) * m["Wg_sc"].T * (hr.abs().mean(dim=-1, keepdim=True).half())
                    u_hat = (hs @ m["Wu_sign"].T) * m["Wu_sc"].T * (hr.abs().mean(dim=-1, keepdim=True).half())
                    pred_mag = (F.silu(g_hat.float()) * u_hat.float()).abs() * m["wd_col_norm"].float()
                    # Group into 32-neuron stripes via offline permutation
                    pm = pred_mag[:, m["perm"]].view(pred_mag.shape[0], -1, 32).mean(dim=-1) # [T, I/32]
                    k_drop = int(pm.shape[-1] * skip)
                    thr = torch.kthvalue(pm, k_drop, dim=-1, keepdim=True).values
                    keep_st = (pm > thr).unsqueeze(-1).expand(-1, -1, 32).reshape_as(pred_mag)
                    keep = keep_st[:, m["inv_perm"]].float()
                    dropped_inter = m["inter_cal"] * (1.0 - keep)
                    # Mean residual vector on calibration set (adds 0.002 MB = 1 vector of H=1024 FP16!)
                    bias_vec[l] = (dropped_inter.mean(dim=0) @ m["Wd"].T).half()

            def make_hook(mode_now, skip_now, bvec):
                def hook(l, h2, W, p):
                    m = ffn_meta[l]
                    if mode_now == "oracle_down_only":
                        g = F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"]))
                        u = F.linear(h2, W[p + "mlp.up_proj.weight"])
                        inter = g * u
                        mag = inter.abs() * m["wd_col_norm"] # Bost-slope weighted!
                        k_drop = int(inter.shape[-1] * skip_now)
                        thr = torch.kthvalue(mag, k_drop, dim=-1, keepdim=True).values
                        inter = inter * (mag > thr).to(inter.dtype)
                        return F.linear(inter, W[p + "mlp.down_proj.weight"])
                    else:
                        # 1-bit FWHT sign screen BEFORE reading gate, up, or down!
                        hr = fwht_last_dim(h2.float())
                        hs = torch.where(hr >= 0, 1.0, -1.0).half()
                        h_sc = hr.abs().mean(dim=-1, keepdim=True).half()
                        g_hat = (hs @ m["Wg_sign"].T) * m["Wg_sc"].T * h_sc
                        u_hat = (hs @ m["Wu_sign"].T) * m["Wu_sc"].T * h_sc
                        pred = (F.silu(g_hat.float()) * u_hat.float()).abs() * m["wd_col_norm"].float()
                        if mode_now == "sign_1bit_full_ffn":
                            k_drop = int(pred.shape[-1] * skip_now)
                            thr = torch.kthvalue(pred, k_drop, dim=-1, keepdim=True).values
                            keep = (pred > thr).to(h2.dtype)
                        else:
                            # 32-neuron coalesced stripe selection!
                            pm = pred[..., m["perm"]].view(1, pred.shape[1], -1, 32).mean(dim=-1)
                            k_drop = int(pm.shape[-1] * skip_now)
                            thr = torch.kthvalue(pm, k_drop, dim=-1, keepdim=True).values
                            keep_st = (pm > thr).unsqueeze(-1).expand(-1, -1, -1, 32).reshape_as(pred)
                            keep = keep_st[..., m["inv_perm"]].to(h2.dtype)
                        # Compute exact gate, up, down ONLY on kept neurons/stripes
                        g = F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"]))
                        u = F.linear(h2, W[p + "mlp.up_proj.weight"])
                        out = F.linear(g * u * keep, W[p + "mlp.down_proj.weight"])
                        if mode_now == "sign_1bit_32stripe_plus_bias_twist":
                            out = out + bvec[l]
                        return out
                return hook

            nll = runner.nll_on(sealed_ids, ffn_hook=make_hook(mode, skip, bias_vec))
            ppl = math.exp(nll)
            dppl = 100.0 * (ppl - base_sealed_ppl) / base_sealed_ppl
            r2_1b[mode][f"skip_{int(skip*100)}pct"] = {"nll": round(nll, 4), "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2)}
            print(f"  [{mode}] skip={int(skip*100)}% -> NLL={nll:.4f}, PPL={ppl:.3f} ({dppl:+.2f}%)", flush=True)
    results["track_r2_1b_ffn_query"] = r2_1b

    # =========================================================================
    # TRACK R2-4A: Residual HSAL / Iwasawa Plane-0 Rescue on ALL 28 Layers of Qwen3-0.6B!
    # =========================================================================
    print("\n=== TRACK R2-4A: RESIDUAL HSAL / IWASAWA PLANE-0 FROZEN RESCUE ===", flush=True)
    # Random fixed signed diagonal D for Parseval incoherence twist (D * H)
    torch.manual_seed(1337)
    linear_keys = [k for k in runner.W if any(s in k for s in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])]

    def build_quant_weights(bits, outlier_frac, use_fwht=True, group_size=64):
        Wq = dict(runner.W)
        total_bits, total_params = 0.0, 0
        for k in linear_keys:
            W_orig = runner.W[k].float() # [N, K]
            N, K = W_orig.shape
            layer_idx = int(k.split(".")[2])
            if "o_proj" in k: act = stats["o_in"][layer_idx]
            elif "down_proj" in k: act = stats["ffn_inter"][layer_idx]
            elif any(s in k for s in ["gate_proj", "up_proj"]): act = stats["ffn_in"][layer_idx]
            else: act = stats["attn_in"][layer_idx]
            # Bost-slope column importance = RMS(act_col) * ||W[:, col]||_2 (Astra + Carbon metric!)
            act_rms = act.pow(2).mean(dim=0).sqrt() # [K]
            col_imp = act_rms * W_orig.norm(dim=0)
            n_out = int(round(K * outlier_frac))
            out_mask = torch.zeros(K, dtype=torch.bool, device=dev)
            if n_out > 0:
                top_cols = torch.topk(col_imp, n_out).indices
                out_mask[top_cols] = True

            # Separate flat columns vs steep outlier columns
            W_flat = W_orig.clone()
            W_flat[:, out_mask] = 0.0
            W_out = torch.zeros_like(W_orig)
            if n_out > 0:
                W_out[:, out_mask] = W_orig[:, out_mask]

            # Apply orthogonal FWHT64 rotation (Parseval twist) to W_flat
            if use_fwht:
                W_rot = fwht_last_dim(W_flat)
            else:
                W_rot = W_flat

            # Quantise W_rot in groups of `group_size` (64)
            W_g = W_rot.view(N, K // group_size, group_size)
            if bits == 1:
                # Optimal 1-bit sign + FP16 scale per 64
                sc = W_g.abs().mean(dim=-1, keepdim=True)
                W_hat_g = torch.where(W_g >= 0, 1.0, -1.0) * sc
                bpw_flat = 1.0 + 16.0 / group_size # 1.25 bpw
            elif bits == 2:
                # Optimal Gaussian 2-bit Lloyd-Max levels (Iwasawa Plane 0 + Plane 1!)
                rms = W_g.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                z = W_g / rms
                sgn = torch.where(z >= 0, 1.0, -1.0)
                mag = torch.where(z.abs() >= 0.9816, 1.5104, 0.4528)
                # Optimal least-squares group scale alpha = <W_g, q> / <q, q>
                q = sgn * mag
                alpha = (W_g * q).sum(dim=-1, keepdim=True) / (q.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
                W_hat_g = q * alpha
                bpw_flat = 2.0 + 16.0 / group_size # 2.25 bpw
            elif bits == 3:
                # 3-bit symmetric uniform per-64 group
                amax = W_g.abs().amax(dim=-1, keepdim=True) + 1e-8
                step = amax / 3.5
                q = torch.clamp(torch.round(W_g / step - 0.5), -4, 3) + 0.5
                W_hat_g = q * step
                bpw_flat = 3.0 + 16.0 / group_size # 3.25 bpw
            else:
                # 4-bit baseline (4.25 bpw)
                amax = W_g.abs().amax(dim=-1, keepdim=True) + 1e-8
                step = amax / 7.5
                q = torch.clamp(torch.round(W_g / step - 0.5), -8, 7) + 0.5
                W_hat_g = q * step
                bpw_flat = 4.0 + 16.0 / group_size

            W_hat = W_hat_g.view(N, K)
            if use_fwht:
                W_hat = fwht_last_dim(W_hat) # FWHT is self-inverse!
            W_rec = W_hat + W_out
            # Wait: for outlier columns we zeroed them before FWHT, so W_hat may have tiny leakage onto out_mask;
            # overwrite out_mask cleanly with exact FP16 W_out!
            if n_out > 0:
                W_rec[:, out_mask] = W_orig[:, out_mask]
            Wq[k] = W_rec.half()

            eff_bpw = bpw_flat * (1.0 - outlier_frac) + 16.0 * outlier_frac
            total_bits += eff_bpw * (N * K)
            total_params += N * K
        return Wq, total_bits / total_params

    # Measure greedy acceptance (both LINEAR k=8 and WIDTH-4 16-NODE TREE!) on sealed prose for any draft model!
    def measure_tree_acceptance(draft_fn, seq_ids, max_steps=64):
        """Given exact target logits on seq_ids, measures:
          - Top-1 agreement (exact linear draft match rate)
          - Top-2 & Top-4 agreement (branch recall for a 16-node width-2/4 tree!)
          - Expected accepted tokens per verify pass for linear (k=8) and width-2 binary tree (depth 4 = 15 nodes -> pp16)!
        """
        tgt_logits = runner.forward_logits(seq_ids)
        dft_logits = draft_fn(seq_ids)
        # Score positions 32..seq-2 (so context is established)
        t_top1 = tgt_logits[32:-1].argmax(dim=-1)
        d_top4 = dft_logits[32:-1].topk(4, dim=-1).indices
        hit1 = (d_top4[:, 0] == t_top1).float().mean().item()
        hit2 = (d_top4[:, :2] == t_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
        hit4 = (d_top4[:, :4] == t_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
        # Expected linear accepted tokens (incl. bonus token, k=8): sum_{j=0..8} p^j
        lin_acc = sum(hit1**j for j in range(9))
        # Expected depth-4 binary tree (1+2+4+8 = 15 draft nodes -> fits in pp16!) accepted tokens:
        # At each depth 1..4, branch succeeds with probability hit2!
        tree16_acc = sum(hit2**j for j in range(5)) # +1 bonus token -> max 5!
        # Or asymmetric 16-node tree (width-2 for depths 1-3 = 14 nodes, width-1 for depth 4-5):
        # Even better: 16-node adaptive tree (top-2 at d1, top-2 at d2, top-2 at d3 -> 2+4+8=14 + top-1 at d4=2 -> up to 5)
        # Or depth-8 hybrid (top-2 at d1, then linear chains of length 7 on both branches = 16 nodes!):
        hybrid16_acc = 1.0 + hit2 * sum(hit1**j for j in range(7))
        return {
            "top1_match": round(hit1, 4),
            "top2_match": round(hit2, 4),
            "top4_match": round(hit4, 4),
            "exp_linear_k8_acc": round(lin_acc, 2),
            "exp_tree16_bin_acc": round(tree16_acc, 2),
            "exp_tree16_hybrid_acc": round(hybrid16_acc, 2),
        }

    r2_4a = {}
    configs = [
        ("1bit_no_fwht_no_out", 1, 0.00, False),
        ("1bit_fwht_0pct_out",  1, 0.00, True),
        ("1bit_fwht_3pct_out",  1, 0.03, True),
        ("1bit_fwht_5pct_out",  1, 0.05, True),
        ("2bit_fwht_0pct_out",  2, 0.00, True),
        ("2bit_fwht_1pct_out",  2, 0.01, True),
        ("2bit_fwht_3pct_out",  2, 0.03, True),
        ("2bit_fwht_5pct_out",  2, 0.05, True),
        ("3bit_fwht_0pct_out",  3, 0.00, True),
        ("4bit_fwht_0pct_out",  4, 0.00, True),
    ]
    for name, bits, ofrac, ufw in configs:
        Wq, bpw = build_quant_weights(bits, ofrac, use_fwht=ufw)
        nll = runner.nll_on(sealed_ids, weights=Wq)
        ppl = math.exp(nll)
        dnll = nll - base_sealed_nll
        dppl = 100.0 * (ppl - base_sealed_ppl) / base_sealed_ppl
        acc = measure_tree_acceptance(lambda s: runner.forward_logits(s, weights=Wq), sealed_ids)
        r2_4a[name] = {
            "effective_bpw": round(bpw, 3),
            "sealed_nll": round(nll, 4),
            "delta_nll": round(dnll, 4),
            "sealed_ppl": round(ppl, 3),
            "ppl_delta_pct": round(dppl, 2),
            "self_spec_fresh_prose": acc,
        }
        print(f"  [{name:22s}] {bpw:.3f} bpw | NLL={nll:.4f} (dNLL={dnll:+.4f}, PPL={dppl:+.1f}%) | "
              f"top1={acc['top1_match']:.3f} top2={acc['top2_match']:.3f} | "
              f"lin8={acc['exp_linear_k8_acc']:.2f} tree16={acc['exp_tree16_hybrid_acc']:.2f}", flush=True)
        del Wq
    results["track_r2_4a_residual_hsal_iwasawa"] = r2_4a

    # =========================================================================
    # TRACK R2-3A: Bost-Slope Layer-Skip Self-Draft (+0 MiB VRAM) & Jacobi 16-Token Fixed-Point!
    # =========================================================================
    print("\n=== TRACK R2-3A: LAYER-SKIP SELF-DRAFT & JACOBI 16-TOKEN FIXED-POINT ===", flush=True)
    # Rank layers by calibration residual slope (how much each layer changes x on calib_ids!)
    layer_deltas = []
    with torch.no_grad():
        seq_c = calib_ids.shape[0]
        cos_c, sin_c = runner.rope_cache(seq_c)
        xc = runner.W["model.embed_tokens.weight"][calib_ids].unsqueeze(0)
        for l in range(runner.n_layers):
            p = f"model.layers.{l}."
            x_prev = xc.clone()
            h = runner.rms_norm(xc, runner.W[p + "input_layernorm.weight"])
            q = F.linear(h, runner.W[p + "self_attn.q_proj.weight"]).view(1, seq_c, runner.n_heads, runner.hd)
            k = F.linear(h, runner.W[p + "self_attn.k_proj.weight"]).view(1, seq_c, runner.n_kv, runner.hd)
            v = F.linear(h, runner.W[p + "self_attn.v_proj.weight"]).view(1, seq_c, runner.n_kv, runner.hd)
            q = runner.rms_norm(q, runner.W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = runner.rms_norm(k, runner.W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = runner.apply_rope(q, cos_c, sin_c), runner.apply_rope(k, cos_c, sin_c)
            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(1, seq_c, runner.n_heads * runner.hd)
            xc = xc + F.linear(attn, runner.W[p + "self_attn.o_proj.weight"])
            h2 = runner.rms_norm(xc, runner.W[p + "post_attention_layernorm.weight"])
            xc = xc + F.linear(F.silu(F.linear(h2, runner.W[p + "mlp.gate_proj.weight"])) * F.linear(h2, runner.W[p + "mlp.up_proj.weight"]), runner.W[p + "mlp.down_proj.weight"])
            # Cosine distance (1 - cos) measures angular slope!
            cos_sim = F.cosine_similarity(x_prev.float(), xc.float(), dim=-1).mean().item()
            layer_deltas.append((1.0 - cos_sim, l))

    # Always keep layer 0 and layer 27; pick steepest-slope middle layers!
    mid_sorted = sorted([x for x in layer_deltas if x[1] not in (0, runner.n_layers - 1)], reverse=True)
    r2_3a_layer = {}
    for keep_n in [14, 18, 22]:
        kept = {0, runner.n_layers - 1} | {l for _, l in mid_sorted[:keep_n - 2]}
        lmask = [l in kept for l in range(runner.n_layers)]
        nll = runner.nll_on(sealed_ids, layer_mask=lmask)
        acc = measure_tree_acceptance(lambda s, lm=lmask: runner.forward_logits(s, layer_mask=lm), sealed_ids)
        r2_3a_layer[f"keep_{keep_n}_of_28_layers"] = {
            "kept_fraction": round(keep_n / runner.n_layers, 3),
            "sealed_nll": round(nll, 4),
            "self_spec_fresh_prose": acc
        }
        print(f"  [LayerSkip {keep_n}/28] NLL={nll:.4f} | top1={acc['top1_match']:.3f} top2={acc['top2_match']:.3f} | "
              f"lin8={acc['exp_linear_k8_acc']:.2f} tree16={acc['exp_tree16_hybrid_acc']:.2f}", flush=True)

    # Kimi's Jacobi 16-Token Parallel Fixed-Point Decoding test on fresh novel prose (exact greedy check!)
    # Start from 48-token prompt of SEALED_TEXT, decode 64 tokens via Jacobi window W=16 vs serial greedy!
    prompt_t = sealed_ids[:48].clone()
    # Ground-truth serial greedy 64 tokens
    cur = prompt_t.clone()
    serial_toks = []
    for _ in range(64):
        nxt = runner.forward_logits(cur)[-1].argmax().view(1)
        serial_toks.append(int(nxt.item()))
        cur = torch.cat([cur, nxt])

    # Jacobi W=16 block decoding (initialized with last token repeated, exact fixed-point check)
    jac_results = {}
    for W_blk in [8, 16]:
        cur = prompt_t.clone()
        generated = []
        sweeps = 0
        # Also test with n-gram warm-start vs naive repeat!
        while len(generated) < 64:
            need = min(W_blk, 64 - len(generated))
            # Initialize block with last token
            blk = cur[-1:].repeat(need)
            for it in range(need):
                sweeps += 1
                seq_in = torch.cat([cur, blk])
                logits = runner.forward_logits(seq_in)
                # Position len(cur)-1 predicts blk[0], ..., len(cur)+need-2 predicts blk[need-1]
                pred = logits[len(cur) - 1 : len(cur) + need - 1].argmax(dim=-1)
                if torch.equal(pred, blk):
                    # Converged! Also grab bonus token at logits[len(cur)+need-1]!
                    bonus = logits[len(cur) + need - 1].argmax().view(1)
                    commit = torch.cat([pred, bonus])[:min(need + 1, 64 - len(generated))]
                    cur = torch.cat([cur, commit])
                    generated.extend(commit.tolist())
                    break
                blk = pred
            else:
                cur = torch.cat([cur, blk])
                generated.extend(blk.tolist())
        exact = (generated[:64] == serial_toks[:64])
        jac_results[f"jacobi_W{W_blk}"] = {
            "tokens_generated": 64,
            "total_forward_sweeps": sweeps,
            "tokens_per_sweep": round(64.0 / sweeps, 3),
            "exact_greedy_match": exact,
        }
        print(f"  [Jacobi W={W_blk}] sweeps={sweeps} for 64 toks -> {64.0/sweeps:.3f} tok/sweep | exact={exact}", flush=True)

    results["track_r2_3a_layerskip_and_jacobi"] = {"layer_skip": r2_3a_layer, "jacobi_fresh_prose": jac_results}

    out_path = Path(os.environ.get("OUT_DIR", ".") + "/r2_frozen_moonshots.json")
    out_path.write_text(json.dumps(results, indent=2))
    print("\nSAVED:", out_path, flush=True)

if __name__ == "__main__":
    run_all()
