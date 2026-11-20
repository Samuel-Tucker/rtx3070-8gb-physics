"""Track R2-2 Real-KV v2.1: Validate Masullo-Shell Sign-KV v2.1 (Exact Top-16 FP16 Spikes + 4-bit / FWHT 2-bit V + Page Centroids)
on REAL 28-Layer Qwen3-0.6B RoPE KV Caches!"""
import json, math, os, re, struct
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

def fwht128(x):
    orig = x.shape
    y = x.reshape(-1, 128).clone()
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2*h), 2, h)
        y = torch.stack([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], dim=-2).reshape(-1, 128)
        h *= 2
    return (y * (1.0 / math.sqrt(128.0))).reshape(orig)

TEXT = (
    "Autonomous robotics requires low-latency perception loops that fuse lidar point clouds with inertial odometry. "
    "During nocturnal navigation across uneven gravel terrain, wheel slip introduces drift that Kalman filters must bound "
    "using sparse visual landmarks. Meanwhile, marine biologists studying cephalopods observe distributed motor control: "
    "each arm contains local ganglia capable of grasping and texture sensing without waiting for central brain commands. "
    "Economic forecasting similarly wrestles with regime shifts where historical correlations break down abruptly during "
    "supply shocks, forcing analysts to weight recent tail observations far more heavily than tranquil averages. "
    "In acoustic engineering, room impulse responses decay exponentially across frequency bands, allowing reverberation "
    "models to split early specular reflections from a diffuse late energy tail. "
    "Structural geologists map fault scarps using airborne laser altimetry to reconstruct prehistoric earthquake cycles "
    "across arid mountain basins where erosion rates remain slow enough to preserve millimetre-scale vertical offsets. "
    "In analytic number theory, decomposing global convolution kernels into smooth coarse shells and local diagonal spikes "
    "controls error in a bare-minimum normed topology while isolating the dominant spectral singularities."
) * 2

def main():
    dev = torch.device("cuda:0")
    cfg = json.loads((SNAP / "config.json").read_text())
    tok_json = json.loads((SNAP / "tokenizer.json").read_text())
    W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
    nl, nh, nkv, hd, eps, theta = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["rope_theta"]
    lm_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])
    ids = torch.tensor(bpe_encode(TEXT, tok_json), device=dev)

    def rms_norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w
    pos = torch.arange(ids.shape[0], device=dev, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    ang = torch.outer(pos, freqs)
    cos_t, sin_t = torch.cos(ang).half()[None, None], torch.sin(ang).half()[None, None]
    def rope(x):
        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return torch.cat([x1*cos_t - x2*sin_t, x2*cos_t + x1*sin_t], dim=-1)

    @torch.no_grad()
    def eval_cfg(mode):
        x = W["model.embed_tokens.weight"][ids].unsqueeze(0)
        S = ids.shape[0]
        cos_all = []
        for l in range(nl):
            p = f"model.layers.{l}."
            h = rms_norm(x, W[p + "input_layernorm.weight"])
            q = rms_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, S, nh, hd), W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = rms_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, S, nkv, hd), W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, S, nkv, hd).transpose(1, 2)
            q, k = rope(q), rope(k)
            kr = k.repeat_interleave(nh // nkv, 1).float()
            vr = v.repeat_interleave(nh // nkv, 1).float()
            qf = q.float()
            ref = F.scaled_dot_product_attention(qf, kr, vr, is_causal=True)

            if mode == "dense_fp16":
                out = ref
            else:
                # 1-bit FWHT K screen
                q_s = torch.where(fwht128(qf) >= 0, 1.0, -1.0)
                k_s = torch.where(fwht128(kr) >= 0, 1.0, -1.0)
                s1 = (q_s @ k_s.transpose(-1, -2)) * (qf.norm(dim=-1, keepdim=True) * kr.norm(dim=-1, keepdim=True).transpose(-1, -2)) * (1.253314 / (hd * math.sqrt(hd)))
                s_ex = (qf @ kr.transpose(-1, -2)) / math.sqrt(hd)
                causal = torch.tril(torch.ones(S, S, device=dev, dtype=torch.bool))
                s1 = s1.masked_fill(~causal, -1e9)

                # Top-16 hottest spikes + 4 sinks + 16 local window use EXACT FP16 K & V (pinned host / ring buffer!)
                idx_r = torch.arange(S, device=dev)
                local_m = (idx_r.unsqueeze(1) - idx_r.unsqueeze(0) >= 0) & (idx_r.unsqueeze(1) - idx_r.unsqueeze(0) < 16)
                sink_m = (idx_r.unsqueeze(0) < 4) & causal

                top16 = s1.topk(min(16, S), dim=-1).indices
                top64 = s1.topk(min(64, S), dim=-1).indices
                ex_m = torch.zeros(1, nh, S, S, device=dev, dtype=torch.bool)
                ex_m.scatter_(-1, top16, True)
                ex_m = (ex_m | local_m | sink_m) & causal

                mid_m = torch.zeros(1, nh, S, S, device=dev, dtype=torch.bool)
                mid_m.scatter_(-1, top64, True)
                mid_m = (mid_m & ~ex_m) & causal

                # Quantise V for mid_m: either 2-bit FWHT-rotated V or 4-bit V
                if "fwht2bitV" in mode:
                    vh = fwht128(vr)
                    rms = vh.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
                    z = vh / rms
                    q2 = torch.where(z >= 0, 1.0, -1.0) * torch.where(z.abs() >= 0.9816, 1.5104, 0.4528)
                    alpha = (vh * q2).sum(dim=-1, keepdim=True) / (q2.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
                    v_mid = fwht128(q2 * alpha)
                else:
                    vmin, vmax = vr.amin(dim=-1, keepdim=True), vr.amax(dim=-1, keepdim=True)
                    step = (vmax - vmin).clamp(min=1e-6) / 15.0
                    v_mid = torch.round((vr - vmin) / step).clamp(0, 15) * step + vmin

                # Masullo 16-token page centroids for background (causal & ~ex_m & ~mid_m)
                P_sz = 16
                npages = (S + P_sz - 1) // P_sz
                pad = npages * P_sz - S
                k_mu = F.pad(kr, (0,0,0,pad)).view(1, nh, npages, P_sz, hd).mean(3).repeat_interleave(P_sz, 2)[:, :, :S, :]
                v_mu = F.pad(vr, (0,0,0,pad)).view(1, nh, npages, P_sz, hd).mean(3).repeat_interleave(P_sz, 2)[:, :, :S, :]
                s_cent = (qf @ k_mu.transpose(-1, -2)) / math.sqrt(hd)

                s_comb = torch.where(ex_m | mid_m, s_ex, s_cent).masked_fill(~causal, -1e9)
                w = F.softmax(s_comb, dim=-1)
                w_ex = w * ex_m.float()
                w_mid = w * mid_m.float()
                w_bg = w * (causal & ~ex_m & ~mid_m).float()
                out = (w_ex @ vr) + (w_mid @ v_mid) + (w_bg @ v_mu)

            cos_all.append(F.cosine_similarity(out, ref, dim=-1).mean().item())
            x = x + F.linear(out.half().transpose(1, 2).contiguous().view(1, S, nh * hd), W[p + "self_attn.o_proj.weight"])
            h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
            x = x + F.linear(F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"])) * F.linear(h2, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])

        x = rms_norm(x, W["model.norm.weight"])
        logits = F.linear(x, lm_w).float().squeeze(0)
        nll = F.cross_entropy(logits[:-1], ids[1:]).item()
        return nll, sum(cos_all) / len(cos_all)

    base_nll, _ = eval_cfg("dense_fp16")
    base_ppl = math.exp(base_nll)
    out = {"tokens": int(ids.numel()), "dense_fp16": {"nll": round(base_nll, 4), "ppl": round(base_ppl, 3)}}
    print(f"[DENSE] tokens={ids.numel()} NLL={base_nll:.4f} PPL={base_ppl:.3f}", flush=True)
    for m in ["v21_exactTop16_fwht2bitV_centroid", "v21_exactTop16_4bitV_centroid"]:
        nll, cos = eval_cfg(m)
        ppl = math.exp(nll)
        dppl = 100.0 * (ppl - base_ppl) / base_ppl
        out[m] = {"28L_mean_cos": round(cos, 5), "nll": round(nll, 4), "ppl": round(ppl, 3), "ppl_delta_pct": round(dppl, 2)}
        print(f"  [{m:36s}] 28L_cos={cos:.5f} | NLL={nll:.4f} PPL={ppl:.3f} ({dppl:+.2f}%)", flush=True)
    Path(os.environ.get("OUT_DIR", ".") + "/r2_real_kv_v21.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
