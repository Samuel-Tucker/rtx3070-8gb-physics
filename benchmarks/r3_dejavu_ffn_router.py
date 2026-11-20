"""Round 3 Build B: Trained Tiny 2-Layer Bost-Slope / DejaVu FFN MLP Router on Frozen Qwen3-0.6B!
Tests whether a tiny 2-layer ReLU MLP router (H -> 128 -> I, ~2.8% of FFN bytes at 8B scale) trained in ~90s
on disjoint training text can predict `|SiLU(gate) * up| * ||W_down[:, i]||_2` BEFORE reading gate, up, or down,
unlocking the 5.22x-7.32x full-FFN INT4 speedup we measured in `coalesced_ffn_v2.cu` on STRICTLY SEALED prose!
"""
import json, math, os, re, struct, time
from pathlib import Path
import torch
import torch.nn as nn
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
    dt_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    return {k: torch.frombuffer(bytearray(raw[m["data_offsets"][0]:m["data_offsets"][1]]), dtype=dt_map[m["dtype"]]).reshape(m["shape"])
            for k, m in header.items() if k != "__metadata__"}

def bpe_encode(text, tok_json):
    vocab, merges = tok_json["model"]["vocab"], tok_json["model"]["merges"]
    rank = {(tuple(m) if isinstance(m, list) else tuple(m.split(" "))): i for i, m in enumerate(merges)}
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    b2u = dict(zip(bs, [chr(c) for c in cs]))
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")
    ids = []
    for tok in pat.findall(text):
        w = tuple(b2u[b] for b in tok.encode("utf-8"))
        while len(w) >= 2:
            best_r, best_i = min((rank.get((w[i], w[i+1]), 10**9), i) for i in range(len(w) - 1))
            if best_r == 10**9: break
            w = w[:best_i] + (w[best_i] + w[best_i+1],) + w[best_i+2:]
        ids.extend(vocab[x] for x in w if x in vocab)
    return ids

# Rich, diverse TRAINING corpus (~1,800 tokens across systems, math, code, biology, economics, history)
TRAIN_TEXT = (
    "Modern semiconductor memory hierarchies separate fast on-chip SRAM from high-capacity off-chip DRAM. "
    "When a processor executes a streaming kernel, cache line alignment and burst coalescing determine effective throughput. "
    "In linear algebra, orthogonal rotations spread energy uniformly across coordinates, reducing peak outlier amplitudes "
    "while preserving inner products by Parseval's identity. Statistical mechanics explains phase transitions in spin lattices "
    "by balancing entropic disorder against local interaction energy. Compilers schedule instruction pipelines to hide load "
    "latencies behind arithmetic operations on independent warp registers. Adaptive quadrature refines only regions where "
    "the derivative exhibits steep local curvature, leaving smooth intervals summarised by low-order moments. "
    "Distributed storage engines replicate write-ahead logs across availability zones to survive correlated power failures "
    "while maintaining strict linearizability for metadata reads. "
    "def compute_sparse_ffn(x, w_gate, w_up, w_down, active_idx):\n"
    "    g = F.silu(F.linear(x, w_gate[active_idx]))\n"
    "    u = F.linear(x, w_up[active_idx])\n"
    "    return F.linear(g * u, w_down[:, active_idx])\n"
    "Photosynthetic organisms capture solar photons using chlorophyll pigment complexes embedded in thylakoid membranes. "
    "Electron transport chains establish proton gradients across the membrane that drive ATP synthase turbines. "
    "In medieval maritime trade, Venetian merchants pioneered double-entry bookkeeping and marine insurance contracts "
    "to share risk across Mediterranean voyages carrying spices, silk, glassware, and timber. "
    "Central banks adjust short-term policy rates to anchor inflation expectations while monitoring labour market slack, "
    "credit spreads, sovereign bond yields, and foreign exchange reserves across open economies. "
    "Aerodynamic wings generate lift through circulation and pressure differentials between upper cambered surfaces and "
    "lower surfaces, while trailing-edge flaps increase maximum lift coefficients during low-speed landing approaches. "
    "Quantum error correction encodes logical qubits into entangled states across many physical qubits so local Pauli errors "
    "can be diagnosed through syndrome measurements without collapsing the encoded quantum information. "
    "Relational database query planners estimate selectivity histograms over B-tree indexes to choose between nested-loop "
    "joins, hash joins, and merge joins under strict working-memory budgets. "
    "Glacial moraines preserve sedimentary records of Pleistocene ice-sheet advances, enabling paleoclimatologists to "
    "correlate oxygen isotope ratios in marine cores with orbital Milankovitch eccentricity cycles. "
    "Modern operating systems isolate user-space processes using hierarchical page tables, translation lookaside buffers, "
    "and hardware privilege rings so a fault in one application cannot corrupt kernel memory or neighbouring services. "
    "In analytic number theory, Dirichlet characters decompose arithmetic progressions into orthogonal harmonic components "
    "whose associated L-functions encode prime distribution irregularities inside the critical strip."
) * 3

# STRICTLY DISJOINT SEALED EVALUATION PROSE (never seen during router training!)
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

class BostRouter(nn.Module):
    """Tiny 2-layer ReLU MLP predictor: H -> R -> I (predicts log(1 + |SiLU(g)*u| * ||W_down||))"""
    def __init__(self, H, I, R=128):
        super().__init__()
        self.fc1 = nn.Linear(H, R, bias=True)
        self.fc2 = nn.Linear(R, I, bias=True)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def main():
    dev = torch.device("cuda:0")
    cfg = json.loads((SNAP / "config.json").read_text())
    tok_json = json.loads((SNAP / "tokenizer.json").read_text())
    W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
    nl, nh, nkv, hd, eps, theta = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["rope_theta"]
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    lm_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])

    train_ids = torch.tensor(bpe_encode(TRAIN_TEXT, tok_json)[:1536], device=dev)
    sealed_ids = torch.tensor(bpe_encode(SEALED_TEXT, tok_json), device=dev)

    def rms_norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    def apply_rope(x, seq):
        pos = torch.arange(seq, device=dev, dtype=torch.float32)
        ang = torch.outer(pos, freqs)
        c, s = torch.cos(ang).half()[None, None], torch.sin(ang).half()[None, None]
        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)

    @torch.no_grad()
    def run_model(ids_t, collect_ffn=False, routers=None, skip=0.0, oracle=False):
        seq = ids_t.shape[0]
        x = W["model.embed_tokens.weight"][ids_t].unsqueeze(0)
        collected = {}
        for l in range(nl):
            p = f"model.layers.{l}."
            h = rms_norm(x, W[p + "input_layernorm.weight"])
            q = rms_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, seq, nh, hd), W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = rms_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, seq, nkv, hd), W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, seq, nkv, hd).transpose(1, 2)
            q, k = apply_rope(q, seq), apply_rope(k, seq)
            attn = F.scaled_dot_product_attention(q, k.repeat_interleave(nh//nkv, 1), v.repeat_interleave(nh//nkv, 1), is_causal=True)
            x = x + F.linear(attn.transpose(1, 2).contiguous().view(1, seq, nh * hd), W[p + "self_attn.o_proj.weight"])

            h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
            Wg = W[p + "mlp.gate_proj.weight"]
            Wu = W[p + "mlp.up_proj.weight"]
            Wd = W[p + "mlp.down_proj.weight"]
            inter = F.silu(F.linear(h2, Wg)) * F.linear(h2, Wu)
            if collect_ffn:
                wd_norm = Wd.float().norm(dim=0)
                imp = inter.squeeze(0).float().abs() * wd_norm
                collected[l] = (h2.squeeze(0).float().cpu(), imp.cpu())
            if skip > 0.0:
                k_drop = int(I * skip)
                if oracle:
                    score = inter.float().abs() * Wd.float().norm(dim=0)
                else:
                    score = routers[l](h2.float())
                thr = torch.kthvalue(score, k_drop, dim=-1, keepdim=True).values
                inter = inter * (score > thr).to(inter.dtype)
            x = x + F.linear(inter, Wd)
        x = rms_norm(x, W["model.norm.weight"])
        logits = F.linear(x, lm_w).float().squeeze(0)
        return F.cross_entropy(logits[:-1], ids_t[1:]).item(), collected

    # 1. Baseline sealed PPL & collect training activations on TRAIN_TEXT
    base_nll, _ = run_model(sealed_ids)
    base_ppl = math.exp(base_nll)
    print(f"[BASE SEALED] tokens={sealed_ids.numel()} NLL={base_nll:.4f} PPL={base_ppl:.3f}", flush=True)
    _, train_data = run_model(train_ids, collect_ffn=True)

    # 2. Train tiny 2-layer BostRouter per layer (R=128 and R=256!)
    results = {
        "train_tokens": int(train_ids.numel()),
        "sealed_tokens": int(sealed_ids.numel()),
        "baseline_dense": {"nll": round(base_nll, 4), "ppl": round(base_ppl, 3)},
        "oracle_down_only": {},
        "trained_routers": {},
    }

    for sp in [0.40, 0.50, 0.60, 0.70]:
        nll, _ = run_model(sealed_ids, skip=sp, oracle=True)
        ppl = math.exp(nll)
        dppl = 100.0 * (ppl - base_ppl) / base_ppl
        results["oracle_down_only"][f"skip_{int(sp*100)}pct"] = {"nll": round(nll, 4), "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2)}
        print(f"  [oracle_down_only ] skip={int(sp*100)}% -> NLL={nll:.4f} PPL={ppl:.3f} ({dppl:+.2f}%)", flush=True)

    for R in [128, 256]:
        t0 = time.perf_counter()
        routers = []
        # Binary top-50% BCE + smooth-L1 magnitude regression (teaches exact ranking boundary!)
        for l in range(nl):
            X_tr, Y_imp = train_data[l][0].to(dev), train_data[l][1].to(dev)
            Y_log = torch.log1p(Y_imp * 10.0)
            # Also binary target for top-50% active neurons
            med = Y_imp.median(dim=-1, keepdim=True).values
            Y_bin = (Y_imp > med).float()

            net = BostRouter(H, I, R=R).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
            N_tr = X_tr.shape[0]
            bs = 256
            for ep in range(35):
                perm = torch.randperm(N_tr, device=dev)
                for b0 in range(0, N_tr, bs):
                    idx = perm[b0:b0+bs]
                    pred = net(X_tr[idx])
                    # Combined ranking/BCE + log-magnitude MSE loss!
                    loss = F.binary_cross_entropy_with_logits(pred, Y_bin[idx]) + 0.5 * F.mse_loss(F.softplus(pred), Y_log[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            routers.append(net)
        train_s = time.perf_counter() - t0
        param_frac_06b = (H * R + R * I) / (3.0 * H * I)
        param_frac_8b  = (4096 * R + R * 12288) / (3.0 * 4096 * 12288)
        print(f"\n=== TRAINED BOST-MLP ROUTER R={R} (28 layers in {train_s:.1f}s | 8B overhead={100*param_frac_8b:.2f}%) ===", flush=True)

        # Measure real GPU wall-clock latency of the router at Qwen3-8B scale (4096 -> R -> 12288 FP16)!
        w1_8b = torch.randn(R, 4096, dtype=torch.float16, device=dev)
        w2_8b = torch.randn(12288, R, dtype=torch.float16, device=dev)
        x_8b = torch.randn(1, 4096, dtype=torch.float16, device=dev)
        for _ in range(20): _ = F.linear(F.relu(F.linear(x_8b, w1_8b)), w2_8b)
        torch.cuda.synchronize()
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(100): _ = F.linear(F.relu(F.linear(x_8b, w1_8b)), w2_8b)
        e_ev.record(); torch.cuda.synchronize()
        router_8b_us = (s_ev.elapsed_time(e_ev) * 1000.0) / 100.0

        r_res = {"train_s": round(train_s, 1), "overhead_pct_8b_ffn": round(100.0 * param_frac_8b, 2), "router_8b_fp16_us": round(router_8b_us, 2), "sealed_eval": {}}
        for sp in [0.40, 0.50, 0.60, 0.70]:
            nll, _ = run_model(sealed_ids, routers=routers, skip=sp, oracle=False)
            ppl = math.exp(nll)
            dppl = 100.0 * (ppl - base_ppl) / base_ppl
            r_res["sealed_eval"][f"skip_{int(sp*100)}pct"] = {"nll": round(nll, 4), "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2)}
            print(f"  [BostRouter R={R:3d}] skip={int(sp*100)}% -> NLL={nll:.4f} PPL={ppl:.3f} ({dppl:+.2f}%) | 8B_router={router_8b_us:.1f} us", flush=True)
        results["trained_routers"][f"R{R}"] = r_res

    out_path = Path(os.environ.get("OUT_DIR", ".")) / "r3_dejavu_ffn_router.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("\nSAVED:", out_path, flush=True)

if __name__ == "__main__":
    main()
