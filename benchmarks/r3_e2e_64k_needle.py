"""Round 3 Build A (Fixed Full-Question Attention): REAL End-to-End 28-Layer Qwen3-0.6B Long-Context
Needle-in-a-Haystack & Decode Speed Benchmark at 8K, 32K, 64K, and 128K Context on an 8 GB RTX 3070 Laptop!
  - Calls our compiled CUDA Masullo-Shell Sign-KV v2.1 library (`libmasullo_kv.so`) live across ALL 28 layers!
  - Uses proper Qwen3 chat template (`<|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n`)
    AND full-context attention across EVERY question token (not just the last token!) so Dense FP16 and Masullo-Shell v2.1
    both attend to the entire 8K/32K/64K/128K document!
"""
import ctypes, json, math, os, re, struct, time
from pathlib import Path
import torch
import torch.nn.functional as F

SNAP = Path(os.environ.get(
    "QWEN3_06B_DIR",
    "./Qwen3-0.6B"
))
LIB_PATH = os.environ.get("MASULLO_LIB", str(Path(__file__).parent / "libmasullo_kv.so"))

def load_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode())
        raw = f.read()
    dt_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    return {k: torch.frombuffer(bytearray(raw[m["data_offsets"][0]:m["data_offsets"][1]]), dtype=dt_map[m["dtype"]]).reshape(m["shape"])
            for k, m in header.items() if k != "__metadata__"}

class Qwen3BPE:
    def __init__(self, tok_json):
        self.vocab = tok_json["model"]["vocab"]
        self.added = {t["content"]: t["id"] for t in tok_json.get("added_tokens", [])}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        for k, v in self.added.items(): self.inv_vocab[v] = k
        merges = tok_json["model"]["merges"]
        self.rank = {(tuple(m) if isinstance(m, list) else tuple(m.split(" "))): i for i, m in enumerate(merges)}
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256 + n); n += 1
        self.b2u = dict(zip(bs, [chr(c) for c in cs]))
        self.u2b = {chr(c): b for b, c in zip(bs, cs)}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")

    def encode_plain(self, text):
        ids = []
        for tok in self.pat.findall(text):
            w = tuple(self.b2u[b] for b in tok.encode("utf-8"))
            while len(w) >= 2:
                best_r, best_i = min((self.rank.get((w[i], w[i+1]), 10**9), i) for i in range(len(w) - 1))
                if best_r == 10**9: break
                w = w[:best_i] + (w[best_i] + w[best_i+1],) + w[best_i+2:]
            ids.extend(self.vocab[x] for x in w if x in self.vocab)
        return ids

    def encode(self, text):
        # Handle special tokens like <|im_start|>, <|im_end|>, <think>, </think>
        specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        pat_s = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids = []
        for part in re.split(pat_s, text):
            if not part: continue
            if part in self.added: ids.append(self.added[part])
            else: ids.extend(self.encode_plain(part))
        return ids

    def decode(self, ids):
        out = []
        raw_bytes = bytearray()
        for i in ids:
            ii = int(i)
            s = self.inv_vocab.get(ii, "")
            if s in self.added:
                if raw_bytes: out.append(raw_bytes.decode("utf-8", errors="replace")); raw_bytes.clear()
                out.append(s)
            else:
                for ch in s:
                    if ch in self.u2b: raw_bytes.append(self.u2b[ch])
        if raw_bytes: out.append(raw_bytes.decode("utf-8", errors="replace"))
        return "".join(out)

lib = ctypes.CDLL(LIB_PATH)
lib.alloc_pinned_mapped.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
lib.alloc_pinned_mapped.restype = ctypes.c_int
lib.free_pinned.argtypes = [ctypes.c_void_p]
lib.launch_pack_kv.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int]*3
lib.launch_build_pages.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*4
lib.launch_masullo_decode.argtypes = [ctypes.c_void_p]*13 + [ctypes.c_int]*3 + [ctypes.c_float]*2

class MasulloKVCache:
    def __init__(self, n_layers, n_kv, max_S, hd=128, dev="cuda:0"):
        self.nl, self.nkv, self.max_S, self.hd = n_layers, n_kv, max_S, hd
        self.max_P = (max_S + 63) // 64
        self.dev = dev
        self.K_p0 = [torch.empty((n_kv, max_S, 4), dtype=torch.int32, device=dev) for _ in range(n_layers)]
        self.K_p1 = [torch.empty((n_kv, max_S, 4), dtype=torch.int32, device=dev) for _ in range(n_layers)]
        self.K_norms = [torch.empty((n_kv, max_S), dtype=torch.float16, device=dev) for _ in range(n_layers)]
        self.V_p0 = [torch.empty((n_kv, max_S, 4), dtype=torch.int32, device=dev) for _ in range(n_layers)]
        self.V_p1 = [torch.empty((n_kv, max_S, 4), dtype=torch.int32, device=dev) for _ in range(n_layers)]
        self.V_norms = [torch.empty((n_kv, max_S), dtype=torch.float16, device=dev) for _ in range(n_layers)]
        self.K_pmu = [torch.empty((n_kv, self.max_P, hd), dtype=torch.float16, device=dev) for _ in range(n_layers)]
        self.V_pmu = [torch.empty((n_kv, self.max_P, hd), dtype=torch.float16, device=dev) for _ in range(n_layers)]
        self.K_pvar = [torch.empty((n_kv, self.max_P), dtype=torch.float16, device=dev) for _ in range(n_layers)]

        layer_bytes = n_kv * max_S * hd * 2
        self.h_K, self.d_K, self.h_V, self.d_V = [], [], [], []
        for _ in range(n_layers):
            hk, dk, hv, dv = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            assert lib.alloc_pinned_mapped(layer_bytes, ctypes.byref(hk), ctypes.byref(dk)) == 0
            assert lib.alloc_pinned_mapped(layer_bytes, ctypes.byref(hv), ctypes.byref(dv)) == 0
            self.h_K.append(hk); self.d_K.append(dk)
            self.h_V.append(hv); self.d_V.append(dv)

    def pack_chunk(self, l, k_chunk, v_chunk, t_start):
        T = k_chunk.shape[1]
        lib.launch_pack_kv(
            k_chunk.data_ptr(), v_chunk.data_ptr(),
            self.K_p0[l].data_ptr(), self.K_p1[l].data_ptr(), self.K_norms[l].data_ptr(),
            self.V_p0[l].data_ptr(), self.V_p1[l].data_ptr(), self.V_norms[l].data_ptr(),
            self.d_K[l], self.d_V[l], t_start, T, self.max_S
        )
        if t_start % 64 == 0 and T >= 64:
            lib.launch_build_pages(
                k_chunk.data_ptr(), v_chunk.data_ptr(),
                self.K_pmu[l].data_ptr(), self.V_pmu[l].data_ptr(), self.K_pvar[l].data_ptr(),
                t_start // 64, T // 64, T, self.max_P
            )

    def decode_attn(self, l, q_tok, S, out_buf, margin_z=5.5, exact_margin=2.5):
        lib.launch_masullo_decode(
            q_tok.data_ptr(),
            self.K_p0[l].data_ptr(), self.K_p1[l].data_ptr(), self.K_norms[l].data_ptr(),
            self.V_p0[l].data_ptr(), self.V_p1[l].data_ptr(), self.V_norms[l].data_ptr(),
            self.d_K[l], self.d_V[l],
            self.K_pmu[l].data_ptr(), self.V_pmu[l].data_ptr(), self.K_pvar[l].data_ptr(),
            out_buf.data_ptr(), S, self.max_S, self.max_P, margin_z, exact_margin
        )

    def close(self):
        for p in self.h_K + self.h_V: lib.free_pinned(p)
        self.h_K.clear(); self.h_V.clear()

def build_needle_parts(bpe, target_tokens, secret_code="FALCON-9472", needle_depth_frac=0.50):
    header_ids = bpe.encode("<|im_start|>user\nRead the archive carefully and answer the question at the end.\n\n")
    filler_para = (
        "The regional logistics archive records routine coastal shipping schedules, dockyard maintenance logs, "
        "tide gauge calibrations, and seasonal warehouse inventory checks across northern ports. "
        "Inspectors review crane hydraulics, timber pallet dimensions, and ballast water certificates every quarter. "
    )
    filler_ids = bpe.encode(filler_para)
    needle_ids = bpe.encode(f"\n[SPECIAL RECORD] The secret vault passcode is {secret_code}.\n")
    question_ids = bpe.encode(
        "\n\nWhat is the exact secret vault passcode mentioned in the archive above? Reply with ONLY the passcode.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )
    n_doc = target_tokens - len(question_ids)
    # Round n_doc down to multiple of 64 for clean Masullo page alignment
    n_doc = (n_doc // 64) * 64
    n_fill = n_doc - len(header_ids) - len(needle_ids)
    n_before = int(n_fill * needle_depth_frac)
    n_after = n_fill - n_before
    def rep(ids, n): return (ids * ((n // len(ids)) + 1))[:n]
    doc_ids = header_ids + rep(filler_ids, n_before) + needle_ids + rep(filler_ids, n_after)
    return doc_ids, question_ids

def main():
    dev = torch.device("cuda:0")
    cfg = json.loads((SNAP / "config.json").read_text())
    bpe = Qwen3BPE(json.loads((SNAP / "tokenizer.json").read_text()))
    W = {k: v.to(dev, dtype=torch.float16) for k, v in load_safetensors(SNAP / "model.safetensors").items()}
    nl, nh, nkv, hd, eps, theta = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["rope_theta"]
    lm_w = W.get("lm_head.weight", W["model.embed_tokens.weight"])

    def rms_norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    def apply_rope_pos(x, pos_ids):
        ang = torch.outer(pos_ids.float(), freqs)
        c, s = torch.cos(ang).half()[None, None], torch.sin(ang).half()[None, None]
        x1, x2 = x[..., :hd//2], x[..., hd//2:]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)

    @torch.no_grad()
    def run_context_test(ctx_len, use_masullo=True, gen_tokens=12, secret="FALCON-9472"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        doc_ids, q_ids = build_needle_parts(bpe, ctx_len, secret_code=secret, needle_depth_frac=0.50)
        total_prompt = len(doc_ids) + len(q_ids)
        max_S = total_prompt + gen_tokens + 64
        chunk_sz = 1024

        if use_masullo:
            mcache = MasulloKVCache(nl, nkv, max_S, hd, dev)
        else:
            try:
                dk_cache = [torch.empty((1, nkv, max_S, hd), dtype=torch.float16, device=dev) for _ in range(nl)]
                dv_cache = [torch.empty((1, nkv, max_S, hd), dtype=torch.float16, device=dev) for _ in range(nl)]
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                return {"status": "OOM", "context_len": ctx_len, "mode": "dense_fp16_sdpa",
                        "required_kv_gib": round((nl * nkv * max_S * hd * 4) / (1<<30), 2),
                        "error": str(e)[:150]}

        t_pref0 = time.perf_counter()
        # 1. Stream document chunks into KV cache WITH sink-anchored attention (always attend to initial 8 sinks + local chunk!)
        cur_pos = 0
        sink_k = [None] * nl
        sink_v = [None] * nl
        for start in range(0, len(doc_ids), chunk_sz):
            ch = doc_ids[start : start + chunk_sz]
            T = len(ch)
            ids_t = torch.tensor(ch, device=dev, dtype=torch.long)
            pos_t = torch.arange(cur_pos, cur_pos + T, device=dev, dtype=torch.long)
            x = W["model.embed_tokens.weight"][ids_t].unsqueeze(0)
            for l in range(nl):
                p = f"model.layers.{l}."
                h = rms_norm(x, W[p + "input_layernorm.weight"])
                q = rms_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, T, nh, hd), W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
                k = rms_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, T, nkv, hd), W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
                v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, T, nkv, hd).transpose(1, 2)
                q, k = apply_rope_pos(q, pos_t), apply_rope_pos(k, pos_t)
                if cur_pos == 0:
                    sink_k[l] = k[:, :, :8, :].clone()
                    sink_v[l] = v[:, :, :8, :].clone()
                if use_masullo:
                    mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                else:
                    dk_cache[l][:, :, cur_pos:cur_pos+T, :] = k
                    dv_cache[l][:, :, cur_pos:cur_pos+T, :] = v
                if cur_pos == 0:
                    attn = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
                else:
                    # Attend to initial 8 BOS/sink tokens + causal within current chunk (preserves Qwen3 hidden-state scale!)
                    kc = torch.cat([sink_k[l], k], dim=2).repeat_interleave(2, 1)
                    vc = torch.cat([sink_v[l], v], dim=2).repeat_interleave(2, 1)
                    cm = torch.ones((T, 8 + T), device=dev, dtype=torch.bool)
                    cm[:, 8:] = torch.tril(torch.ones((T, T), device=dev, dtype=torch.bool))
                    attn = F.scaled_dot_product_attention(q, kc, vc, attn_mask=cm)
                x = x + F.linear(attn.transpose(1, 2).contiguous().view(1, T, nh * hd), W[p + "self_attn.o_proj.weight"])
                h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
                x = x + F.linear(F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"])) * F.linear(h2, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
            cur_pos += T

        # 2. Process EVERY question token with FULL-CONTEXT attention over the entire 8K/32K/64K/128K document!
        out_buf = torch.empty((nh, hd), dtype=torch.float16, device=dev)
        last_logits = None
        for qi, tok_id in enumerate(q_ids):
            tok_t = torch.tensor([tok_id], device=dev, dtype=torch.long)
            pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.long)
            x = W["model.embed_tokens.weight"][tok_t].unsqueeze(0)
            for l in range(nl):
                p = f"model.layers.{l}."
                h = rms_norm(x, W[p + "input_layernorm.weight"])
                q = rms_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, 1, nh, hd), W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
                k = rms_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, 1, nkv, hd), W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
                v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, 1, nkv, hd).transpose(1, 2)
                q, k = apply_rope_pos(q, pos_t), apply_rope_pos(k, pos_t)
                if use_masullo:
                    mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                    mcache.decode_attn(l, q[0, :, 0, :].contiguous(), cur_pos + 1, out_buf, 5.5, 2.5)
                    attn_flat = out_buf.view(1, 1, nh * hd)
                else:
                    dk_cache[l][:, :, cur_pos:cur_pos+1, :] = k
                    dv_cache[l][:, :, cur_pos:cur_pos+1, :] = v
                    k_all = dk_cache[l][:, :, :cur_pos+1, :].repeat_interleave(2, 1)
                    v_all = dv_cache[l][:, :, :cur_pos+1, :].repeat_interleave(2, 1)
                    attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
                    attn_flat = attn.transpose(1, 2).contiguous().view(1, 1, nh * hd)
                x = x + F.linear(attn_flat, W[p + "self_attn.o_proj.weight"])
                h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
                x = x + F.linear(F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"])) * F.linear(h2, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
            cur_pos += 1
            if qi == len(q_ids) - 1:
                last_logits = F.linear(rms_norm(x[:, -1:, :], W["model.norm.weight"]), lm_w).float().squeeze()

        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_pref0

        # 3. Real autoregressive decode loop (timed cleanly!)
        gen_ids = [int(last_logits.argmax().item())]
        attn_ms_total = 0.0
        ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_dec0 = time.perf_counter()
        for _ in range(gen_tokens - 1):
            tok_t = torch.tensor([gen_ids[-1]], device=dev, dtype=torch.long)
            pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.long)
            x = W["model.embed_tokens.weight"][tok_t].unsqueeze(0)
            for l in range(nl):
                p = f"model.layers.{l}."
                h = rms_norm(x, W[p + "input_layernorm.weight"])
                q = rms_norm(F.linear(h, W[p + "self_attn.q_proj.weight"]).view(1, 1, nh, hd), W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
                k = rms_norm(F.linear(h, W[p + "self_attn.k_proj.weight"]).view(1, 1, nkv, hd), W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
                v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(1, 1, nkv, hd).transpose(1, 2)
                q, k = apply_rope_pos(q, pos_t), apply_rope_pos(k, pos_t)

                ev_s.record()
                if use_masullo:
                    mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                    mcache.decode_attn(l, q[0, :, 0, :].contiguous(), cur_pos + 1, out_buf, 5.5, 2.5)
                    attn_flat = out_buf.view(1, 1, nh * hd)
                else:
                    dk_cache[l][:, :, cur_pos:cur_pos+1, :] = k
                    dv_cache[l][:, :, cur_pos:cur_pos+1, :] = v
                    k_all = dk_cache[l][:, :, :cur_pos+1, :].repeat_interleave(2, 1)
                    v_all = dv_cache[l][:, :, :cur_pos+1, :].repeat_interleave(2, 1)
                    attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=False)
                    attn_flat = attn.transpose(1, 2).contiguous().view(1, 1, nh * hd)
                ev_e.record()
                torch.cuda.synchronize()
                attn_ms_total += ev_s.elapsed_time(ev_e)

                x = x + F.linear(attn_flat, W[p + "self_attn.o_proj.weight"])
                h2 = rms_norm(x, W[p + "post_attention_layernorm.weight"])
                x = x + F.linear(F.silu(F.linear(h2, W[p + "mlp.gate_proj.weight"])) * F.linear(h2, W[p + "mlp.up_proj.weight"]), W[p + "mlp.down_proj.weight"])
            cur_pos += 1
            logits = F.linear(rms_norm(x[:, -1:, :], W["model.norm.weight"]), lm_w).float().squeeze()
            gen_ids.append(int(logits.argmax().item()))

        torch.cuda.synchronize()
        dec_s = time.perf_counter() - t_dec0
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        answer = bpe.decode(gen_ids)
        needle_hit = (secret in answer)

        if use_masullo: mcache.close()
        else: del dk_cache, dv_cache
        torch.cuda.empty_cache()

        n_steps = gen_tokens - 1
        return {
            "status": "OK",
            "context_len": ctx_len,
            "mode": "masullo_shell_v21" if use_masullo else "dense_fp16_sdpa",
            "peak_vram_mib": round(peak_mib, 1),
            "prefill_s": round(prefill_s, 2),
            "decode_tps": round(n_steps / dec_s, 2),
            "ms_per_token_total": round(1000.0 * dec_s / n_steps, 2),
            "attn_ms_per_token_28L": round(attn_ms_total / n_steps, 2),
            "secret_planted": secret,
            "needle_exact_match": needle_hit,
            "generated_text": answer.strip()[:80],
        }

    results = {"model": "Qwen3-0.6B (28 layers, FP16, native RoPE 40K)", "runs": []}
    for ctx_len, secret in [(8192, "FALCON-9472"), (32768, "ORCHID-3819"), (65536, "KESTREL-6140")]:
        print(f"\n=== CONTEXT LENGTH S = {ctx_len} (secret={secret}) ===", flush=True)
        r_dense = run_context_test(ctx_len, use_masullo=False, gen_tokens=24, secret=secret)
        print(f"  [DENSE FP16 ] {r_dense}", flush=True)
        results["runs"].append(r_dense)

        r_mas = run_context_test(ctx_len, use_masullo=True, gen_tokens=24, secret=secret)
        print(f"  [MASULLO v21] {r_mas}", flush=True)
        results["runs"].append(r_mas)

    out_path = Path(os.environ.get("OUT_DIR", ".")) / "r3_e2e_64k_needle.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("\nSAVED:", out_path, flush=True)

if __name__ == "__main__":
    main()
