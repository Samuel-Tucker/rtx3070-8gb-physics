import os
"""Track 5: Opus 5.5 H3 (Zero-Training Activation Column Skipping on Qwen3-0.6B)
+ Fable 5.2 Killer Pick (Multi-Stream Aggregate Throughput & Qwen3-4B/8B ngram-simple Speculation)."""
import json, math, os, struct, subprocess, time, urllib.request
from pathlib import Path
import torch
import torch.nn.functional as F

SNAP = Path(os.environ.get("QWEN3_06B_DIR", "./Qwen3-0.6B"))

def load_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode())
        base = 8 + hlen
        raw = f.read()
    tensors = {}
    dt_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    for k, meta in header.items():
        if k == "__metadata__": continue
        s, e = meta["data_offsets"]
        t = torch.frombuffer(bytearray(raw[s:e]), dtype=dt_map[meta["dtype"]]).reshape(meta["shape"])
        tensors[k] = t
    return tensors

def run_opus_h3_sparsity():
    cfg = json.loads((SNAP / "config.json").read_text())
    tok_json = json.loads((SNAP / "tokenizer.json").read_text())
    vocab = tok_json["model"]["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}

    # Simple greedy longest-match tokenizer over GPT-2 byte-level BPE vocab for held-out NLL eval
    text = (
        "The NVIDIA Ampere architecture organizes compute into streaming multiprocessors. "
        "Each streaming multiprocessor contains warp schedulers, tensor cores, and shared memory. "
        "During autoregressive language model decoding, reading dense weights from GDDR6 memory dominates wall-clock latency "
        "because single-token matrix-vector multiplication has an arithmetic intensity of roughly one flop per byte. "
        "def benchmark_matmul(n, k, batch_size):\n"
        "    weights = torch.randn(n, k, dtype=torch.float16, device='cuda')\n"
        "    inputs = torch.randn(batch_size, k, dtype=torch.float16, device='cuda')\n"
        "    return torch.matmul(inputs, weights.T)\n"
        "In number theory, the distribution of prime numbers connects to zeros of the Riemann zeta function on the critical line. "
        "By composing short structured lifts instead of monolithic transformations, both algorithms and GPU kernels can bypass classic bottlenecks."
    ) * 3
    # Encode with Ġ for spaces
    s_enc = text.replace(" ", "Ġ").replace("\n", "Ċ")
    ids = []
    i = 0
    while i < len(s_enc):
        matched = False
        for L in range(min(16, len(s_enc) - i), 0, -1):
            sub = s_enc[i:i+L]
            if sub in vocab:
                ids.append(vocab[sub]); i += L; matched = True; break
        if not matched:
            ids.append(vocab.get(s_enc[i], 0)); i += 1
    ids = ids[:384]

    dev = torch.device("cuda:0")
    W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]

    def rms_norm(x, weight):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight

    seq = len(ids)
    pos = torch.arange(seq, device=dev, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    ang = torch.outer(pos, freqs)
    cos_t = torch.cos(ang).to(torch.float16)[None, None, :, :]
    sin_t = torch.sin(ang).to(torch.float16)[None, None, :, :]

    def apply_rope(x):
        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

    def eval_nll(sparsity=0.0):
        x = W["model.embed_tokens.weight"][torch.tensor(ids, device=dev)].unsqueeze(0)
        for l in range(n_layers):
            p = f"model.layers.{l}."
            h = rms_norm(x, W[p + "input_layernorm.weight"])
            q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, seq, n_heads, hd)
            k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, seq, n_kv, hd)
            v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, seq, n_kv, hd)
            q = rms_norm(q, W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
            k = rms_norm(k, W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = apply_rope(q), apply_rope(k)
            k_rep = k.repeat_interleave(n_heads // n_kv, dim=1)
            v_rep = v.repeat_interleave(n_heads // n_kv, dim=1)
            attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(1, seq, n_heads * hd)
            x = x + F.linear(attn, W[p + "self_attn.o_proj.weight"])

            h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
            g = F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"]))
            u = F.linear(h2, W[p + "mlp.up_proj.weight"])
            inter = g * u
            if sparsity > 0.0:
                # Opus 5.5 H3: magnitude-gated channel pruning on FFN intermediate (down_proj columns)
                mag = inter.abs()
                k_drop = int(inter.shape[-1] * sparsity)
                if k_drop > 0:
                    thresh = torch.kthvalue(mag, k_drop, dim=-1, keepdim=True).values
                    inter = inter * (mag > thresh).to(inter.dtype)
            x = x + F.linear(inter, W[p + "mlp.down_proj.weight"])
        x = rms_norm(x, W["model.norm.weight"])
        lm_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])
        logits = F.linear(x[:, :-1, :], lm_w).float()
        targets = torch.tensor(ids[1:], device=dev)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets).item()

    res = {}
    with torch.no_grad():
        base_nll = eval_nll(0.0)
        res["dense_fp16"] = {"sparsity": 0.0, "nll": round(base_nll, 4), "ppl": round(math.exp(base_nll), 3), "ppl_delta_pct": 0.0}
        for s in [0.25, 0.40, 0.50, 0.60, 0.70]:
            nll = eval_nll(s)
            ppl = math.exp(nll)
            bppl = math.exp(base_nll)
            res[f"skip_{int(s*100)}pct"] = {
                "sparsity": s,
                "nll": round(nll, 4),
                "ppl": round(ppl, 3),
                "ppl_delta_pct": round(100.0 * (ppl - bppl) / bppl, 2),
            }
    return {"tokens_scored": len(ids), "opus_h3_ffn_skipping": res}

def run_fable_and_4b_ngram():
    BIN = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
    M4 = os.environ.get("QWEN3_4B_GGUF", "./Qwen3-4B-Q4_K_M.gguf")
    M8 = os.environ.get("QWEN3_8B_GGUF", "./Qwen3-8B-Q4_K_M.gguf")
    PROMPT_CODE = """Here is a Python CUDA benchmark script. Refactor it to add FP8/INT4 quantization timing, keep all existing functions, comments, and field names intact, and return the complete updated script:

```python
import time, torch, json

def bench_ms(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def run_roofline(dev):
    results = {}
    for kb in [128, 512, 1536, 3072, 6144, 16384, 65536]:
        n = (kb * 1024) // 4
        a = torch.empty(n, dtype=torch.float32, device=dev)
        b = torch.empty(n, dtype=torch.float32, device=dev)
        ms = bench_ms(lambda: b.copy_(a), warmup=20, iters=60)
        results[f"{kb}KB"] = round((2 * kb * 1024) / (ms * 1e-3) / 1e9, 1)
    return results
```
Return the complete refactored Python script now:
```python
"""
    def test_srv(name, args):
        port = 18923
        cmd = [BIN, "--port", str(port), "--host", "127.0.0.1", "-ngl", "99", "-fa", "on", "-c", "4096"] + args
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            t0 = time.time()
            while time.time() - t0 < 45:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200: break
                except Exception: pass
                time.sleep(0.3)
            def ask(pr, np=256):
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                    data=json.dumps({"prompt": pr, "n_predict": np, "temperature": 0.0, "top_k": 1, "seed": 42, "cache_prompt": False}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read().decode())
                t = d.get("timings", {})
                return {"tps": round(t.get("predicted_per_second", 0), 2), "draft_n": t.get("draft_n"), "acc": t.get("draft_n_accepted"), "text": d.get("content","")}
            _ = ask("Hello", 16)
            out = ask(PROMPT_CODE, 256)
            return out
        finally:
            p.terminate()
            try: p.wait(timeout=8)
            except Exception: p.kill()
            time.sleep(0.8)

    # 1. Qwen3-4B ngram-simple vs serial (fixing ngram-mod not triggering!)
    r4_ser = test_srv("4b_serial", ["-m", M4, "--spec-type", "none"])
    r4_ng15 = test_srv("4b_ngram_simple_15", ["-m", M4, "--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "6", "--spec-ngram-simple-size-m", "15", "--spec-draft-n-max", "15"])
    r4_ng31 = test_srv("4b_ngram_simple_31", ["-m", M4, "--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "6", "--spec-ngram-simple-size-m", "31", "--spec-draft-n-max", "31"])
    r8_ng31 = test_srv("8b_ngram_simple_31", ["-m", M8, "--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "6", "--spec-ngram-simple-size-m", "31", "--spec-draft-n-max", "31"])

    # 2. Fable 5.2 Killer Pick: Multi-stream parallel decode on Qwen3-4B (1, 4, 16, 32 concurrent slots!)
    import concurrent.futures
    multi = {}
    for np_slots in [1, 4, 16, 32]:
        port = 18924
        cmd = [BIN, "--port", str(port), "--host", "127.0.0.1", "-ngl", "99", "-fa", "on", "-c", str(max(4096, np_slots * 512)), "-np", str(np_slots), "-m", M4]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            t0 = time.time()
            while time.time() - t0 < 45:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200: break
                except Exception: pass
                time.sleep(0.3)
            def one_req(i):
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                    data=json.dumps({"prompt": f"Explain concept #{i} in computer architecture concisely:", "n_predict": 64, "temperature": 0.0, "cache_prompt": False}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode()).get("timings", {}).get("predicted_n", 64)
            # warmup 1
            _ = one_req(0)
            ts = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=np_slots) as ex:
                toks = sum(ex.map(one_req, range(1, np_slots + 1)))
            dt = time.perf_counter() - ts
            multi[f"streams_{np_slots}"] = {"total_tokens": toks, "wall_s": round(dt, 3), "agg_tps": round(toks / dt, 1), "per_stream_tps": round(toks / dt / np_slots, 1)}
        finally:
            p.terminate()
            try: p.wait(timeout=8)
            except Exception: p.kill()
            time.sleep(0.8)

    return {
        "qwen3_4b_serial_code_tps": r4_ser["tps"],
        "qwen3_4b_ngram_simple_k15": {"tps": r4_ng15["tps"], "speedup": round(r4_ng15["tps"]/r4_ser["tps"], 2), "draft_n": r4_ng15["draft_n"], "acc": r4_ng15["acc"], "exact_match": r4_ng15["text"] == r4_ser["text"]},
        "qwen3_4b_ngram_simple_k31": {"tps": r4_ng31["tps"], "speedup": round(r4_ng31["tps"]/r4_ser["tps"], 2), "draft_n": r4_ng31["draft_n"], "acc": r4_ng31["acc"], "exact_match": r4_ng31["text"] == r4_ser["text"]},
        "qwen3_8b_ngram_simple_k31": {"tps": r8_ng31["tps"], "draft_n": r8_ng31["draft_n"], "acc": r8_ng31["acc"]},
        "fable_parallel_streams_qwen3_4b": multi,
    }

if __name__ == "__main__":
    out = {"opus_h3": run_opus_h3_sparsity(), "fable_and_ngram": run_fable_and_4b_ngram()}
    with open("track5_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
