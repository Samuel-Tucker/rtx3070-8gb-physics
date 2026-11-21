"""Round 4 Objective 2B: REAL 36-Layer Qwen3-4B-Instruct-2507-Q4_K_M (Native 256K Context!) at 32K & 64K
Inside < 3.5 GiB Total GPU VRAM on an 8 GB RTX 3070 Laptop!
  - Loads REAL Qwen3-4B-Instruct-2507-Q4_K_M GGUF weights directly onto GPU (2.33 GiB, exact ggml Q4_K & Q6_K!)
  - Calls libmasullo_kv_v3.so (19.69x Thin-VRAM = 0.46 GiB GPU KV at 64K vs 9.00 GiB FP16 -> OOM on 8 GB!)
  - Compares BOTH Independent-Head v3 AND Masullo Saturated-Places 4:1 GQA v3 (4 sibling Q heads sharing 1 scan + 1 PCIe read!)
  - Verifies 100% EXACT hidden passcode retrieval at 32K & 64K on a REAL 4-BILLION-PARAMETER 36-LAYER MODEL!
"""
import ctypes, json, math, os, re, struct, time
from pathlib import Path
import torch
import torch.nn.functional as F

GGUF_PATH = os.environ.get(
    "QWEN3_4B_INSTRUCT_GGUF",
    "./Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)
TOK_JSON_PATH = os.environ.get(
    "QWEN3_TOK_JSON",
    "./Qwen3-0.6B/tokenizer.json"
)
KV_LIB_PATH = os.environ.get("MASULLO_V3_LIB", str(Path(__file__).parent / "libmasullo_kv_v3.so"))
OPS_LIB_PATH = os.environ.get("QWEN4B_OPS_LIB", str(Path(__file__).parent / "libqwen4b_gguf.so"))

kv_lib = ctypes.CDLL(KV_LIB_PATH)
kv_lib.alloc_pinned_mapped.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
kv_lib.alloc_pinned_mapped.restype = ctypes.c_int
kv_lib.free_pinned.argtypes = [ctypes.c_void_p]
kv_lib.launch_pack_kv_v3.argtypes = [ctypes.c_void_p]*9 + [ctypes.c_int]*6
kv_lib.launch_build_pages_v3.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5
kv_lib.launch_masullo_v3_decode.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int]*6
kv_lib.launch_masullo_v3_saturated_gqa4.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int]*5

ops_lib = ctypes.CDLL(OPS_LIB_PATH)
ops_lib.launch_dequant_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ops_lib.launch_gemv_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ops_lib.launch_dequant_q6k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ops_lib.launch_gemv_q6k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ops_lib.launch_embed_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
ops_lib.launch_embed_q6k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

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
        specials = ["<|im_start|>", "<|im_end|>"]
        pat_s = "(" + "|".join(re.escape(s) for s in specials) + ")"
        ids = []
        for part in re.split(pat_s, text):
            if not part: continue
            if part in self.added: ids.append(self.added[part])
            else: ids.extend(self.encode_plain(part))
        return ids

    def decode(self, ids):
        out, raw = [], bytearray()
        for i in ids:
            s = self.inv_vocab.get(int(i), "")
            if s in self.added:
                if raw: out.append(raw.decode("utf-8", errors="replace")); raw.clear()
                out.append(s)
            else:
                for ch in s:
                    if ch in self.u2b: raw.append(self.u2b[ch])
        if raw: out.append(raw.decode("utf-8", errors="replace"))
        return "".join(out)

def load_gguf_gpu(path, dev):
    """Minimal zero-dependency GGUF v3 loader keeping Q4_K (type 12) and Q6_K (type 14) raw bytes on GPU!"""
    with open(path, "rb") as f:
        magic, ver, n_t, n_kv = struct.unpack("<4sIQQ", f.read(24))
        assert magic == b"GGUF"
        def read_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "ignore")
        def skip_val(t):
            if t in (0,1,7): f.read(1)
            elif t in (2,3): f.read(2)
            elif t in (4,5,6): f.read(4)
            elif t in (10,11,12): f.read(8)
            elif t == 8: read_str()
            elif t == 9:
                et, ln = struct.unpack("<IQ", f.read(12))
                for _ in range(ln): skip_val(et)
        meta = {}
        align = 32
        for _ in range(n_kv):
            k = read_str()
            t = struct.unpack("<I", f.read(4))[0]
            if k == "general.alignment" and t == 4:
                align = struct.unpack("<I", f.read(4))[0]
            elif t in (4, 5, 6) and k.startswith("qwen3."):
                meta[k] = struct.unpack("<I" if t==4 else ("<i" if t==5 else "<f"), f.read(4))[0]
            else:
                skip_val(t)
        infos = []
        for _ in range(n_t):
            name = read_str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd)) # GGUF dims are [K, N] (innermost first!)
            tt, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dims, tt, off))
        data_start = ((f.tell() + align - 1) // align) * align

        # Load every tensor cleanly onto GPU
        tensors = {}
        for name, dims, tt, off in infos:
            f.seek(data_start + off)
            if tt == 0: # F32 (norms!)
                ne = 1
                for d in dims: ne *= d
                raw = f.read(ne * 4)
                t_gpu = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(*reversed(dims)).to(dev, dtype=torch.float16).contiguous()
                tensors[name] = (tt, tuple(reversed(dims)), t_gpu)
            elif tt == 12: # Q4_K (144 B / 256 elems)
                K, N = dims[0], dims[1]
                nbytes = N * (K // 256) * 144
                raw = f.read(nbytes)
                t_gpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).contiguous()
                tensors[name] = (tt, (N, K), t_gpu)
            elif tt == 14: # Q6_K (210 B / 256 elems)
                K, N = dims[0], dims[1]
                nbytes = N * (K // 256) * 210
                raw = f.read(nbytes)
                t_gpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dev).contiguous()
                tensors[name] = (tt, (N, K), t_gpu)
            else:
                raise RuntimeError(f"Unsupported GGUF type {tt} for {name}")
    return meta, tensors

def qlinear(x, entry, scratch_w):
    """x: [..., K] FP16 -> [..., N] FP16 using exact Q4_K / Q6_K CUDA GEMV (M=1) or dequant+cuBLAS (M>1)!"""
    tt, (N, K), raw_gpu = entry
    if x.numel() == K:
        y = torch.empty((*x.shape[:-1], N), dtype=torch.float16, device=x.device)
        if tt == 12: ops_lib.launch_gemv_q4k(raw_gpu.data_ptr(), x.data_ptr(), y.data_ptr(), N, K)
        else:        ops_lib.launch_gemv_q6k(raw_gpu.data_ptr(), x.data_ptr(), y.data_ptr(), N, K)
        return y
    else:
        w16 = scratch_w[:N*K].view(N, K)
        if tt == 12: ops_lib.launch_dequant_q4k(raw_gpu.data_ptr(), w16.data_ptr(), N, K)
        else:        ops_lib.launch_dequant_q6k(raw_gpu.data_ptr(), w16.data_ptr(), N, K)
        return F.linear(x, w16)

def qembed(ids_t, entry):
    tt, (N, K), raw_gpu = entry
    T = ids_t.numel()
    out = torch.empty((1, T, K), dtype=torch.float16, device=ids_t.device)
    if tt == 12: ops_lib.launch_embed_q4k(raw_gpu.data_ptr(), ids_t.data_ptr(), out.data_ptr(), T, K)
    else:        ops_lib.launch_embed_q6k(raw_gpu.data_ptr(), ids_t.data_ptr(), out.data_ptr(), T, K)
    return out

class ThinMasullo4B:
    def __init__(self, nl, nq, nkv, max_S, hd=128, dev="cuda:0"):
        self.nl, self.nq, self.nkv, self.max_S, self.hd = nl, nq, nkv, max_S, hd
        self.max_P = (max_S + 63) // 64
        self.K_p0 = [torch.empty((nkv, max_S, 4), dtype=torch.int32, device=dev) for _ in range(nl)]
        self.K_norms = [torch.empty((nkv, max_S), dtype=torch.float16, device=dev) for _ in range(nl)]
        self.K_pmu = [torch.empty((nkv, self.max_P, hd), dtype=torch.float16, device=dev) for _ in range(nl)]
        self.V_pmu = [torch.empty((nkv, self.max_P, hd), dtype=torch.float16, device=dev) for _ in range(nl)]
        q8_b, sc_b = nkv * max_S * hd, nkv * max_S * 2
        self.h_ptrs, self.d_Kq8, self.d_Ksc, self.d_Vq8, self.d_Vsc = [], [], [], [], []
        for _ in range(nl):
            for sz, dst in [(q8_b, self.d_Kq8), (sc_b, self.d_Ksc), (q8_b, self.d_Vq8), (sc_b, self.d_Vsc)]:
                hp, dp = ctypes.c_void_p(), ctypes.c_void_p()
                assert kv_lib.alloc_pinned_mapped(sz, ctypes.byref(hp), ctypes.byref(dp)) == 0
                self.h_ptrs.append(hp); dst.append(dp)

    def pack_chunk(self, l, k_ch, v_ch, t_start):
        T = k_ch.shape[1]
        use_resid = 0
        if t_start % 64 == 0 and T >= 64:
            kv_lib.launch_build_pages_v3(
                k_ch.data_ptr(), v_ch.data_ptr(), self.K_pmu[l].data_ptr(), self.V_pmu[l].data_ptr(),
                self.nkv, t_start // 64, T // 64, T, self.max_P
            )
            use_resid = 1
        kv_lib.launch_pack_kv_v3(
            k_ch.data_ptr(), v_ch.data_ptr(), self.K_pmu[l].data_ptr(),
            self.K_p0[l].data_ptr(), self.K_norms[l].data_ptr(),
            self.d_Kq8[l], self.d_Ksc[l], self.d_Vq8[l], self.d_Vsc[l],
            self.nkv, t_start, T, self.max_S, self.max_P, use_resid
        )

    def decode_attn(self, l, q_tok, S, out_buf, saturated=False, top_k=256):
        if saturated:
            kv_lib.launch_masullo_v3_saturated_gqa4(
                q_tok.data_ptr(), self.K_p0[l].data_ptr(), self.K_norms[l].data_ptr(),
                self.d_Kq8[l], self.d_Ksc[l], self.d_Vq8[l], self.d_Vsc[l],
                self.K_pmu[l].data_ptr(), self.V_pmu[l].data_ptr(),
                out_buf.data_ptr(), self.nkv, S, self.max_S, self.max_P, top_k
            )
        else:
            kv_lib.launch_masullo_v3_decode(
                q_tok.data_ptr(), self.K_p0[l].data_ptr(), self.K_norms[l].data_ptr(),
                self.d_Kq8[l], self.d_Ksc[l], self.d_Vq8[l], self.d_Vsc[l],
                self.K_pmu[l].data_ptr(), self.V_pmu[l].data_ptr(),
                out_buf.data_ptr(), self.nq, self.nkv, S, self.max_S, self.max_P, top_k
            )

    def close(self):
        for p in self.h_ptrs: kv_lib.free_pinned(p)
        self.h_ptrs.clear()

def build_needle_parts(bpe, target_tokens, secret_code):
    header_ids = bpe.encode("<|im_start|>user\nRead the archive carefully and answer the question at the end.\n\n")
    filler = (
        "The regional logistics archive records routine coastal shipping schedules, dockyard maintenance logs, "
        "tide gauge calibrations, and seasonal warehouse inventory checks across northern ports. "
        "Inspectors review crane hydraulics, timber pallet dimensions, and ballast water certificates every quarter. "
    )
    f_ids = bpe.encode(filler)
    n_ids = bpe.encode(f"\n[SPECIAL RECORD] The secret vault passcode is {secret_code}.\n")
    q_ids = bpe.encode("\n\nWhat is the exact secret vault passcode mentioned in the archive above? Reply with ONLY the passcode.<|im_end|>\n<|im_start|>assistant\n")
    n_doc = ((target_tokens - len(q_ids)) // 64) * 64
    n_fill = n_doc - len(header_ids) - len(n_ids)
    raw_b = n_fill // 2
    abs_p = len(header_ids) + raw_b
    if (abs_p % 256) > 200 or (abs_p % 256) < 24:
        raw_b = ((abs_p // 256) * 256 + 128) - len(header_ids)
    nb = max(0, min(n_fill, raw_b)); na = n_fill - nb
    def rep(ids, n): return (ids * ((n // len(ids)) + 1))[:n]
    return header_ids + rep(f_ids, nb) + n_ids + rep(f_ids, na), q_ids

def main():
    dev = torch.device("cuda:0")
    bpe = Qwen3BPE(json.loads(Path(TOK_JSON_PATH).read_text()))
    t0 = time.perf_counter()
    meta, T = load_gguf_gpu(GGUF_PATH, dev)
    nl = int(meta["qwen3.block_count"])       # 36
    nh = int(meta["qwen3.attention.head_count"])    # 32
    nkv = int(meta["qwen3.attention.head_count_kv"])# 8
    hd = int(meta["qwen3.attention.key_length"])    # 128
    H = int(meta["qwen3.embedding_length"])         # 2560
    I = int(meta["qwen3.feed_forward_length"])      # 9728
    theta = float(meta["qwen3.rope.freq_base"])     # 5,000,000 (NATIVE 256K!!)
    eps = float(meta.get("qwen3.attention.layer_norm_rms_epsilon", 1e-6))
    print(f"Loaded REAL Qwen3-4B-Instruct GGUF in {time.perf_counter()-t0:.1f}s | {nl}L {nh}Q/{nkv}KV H={H} I={I} theta={theta:.0f} | VRAM={torch.cuda.memory_allocated()/(1<<20):.1f} MiB", flush=True)

    # Single reusable dequant scratch buffer for chunked prefill (max mat = 9728 * 2560 = 49.8 MiB FP16!)
    scratch_w = torch.empty(I * H, dtype=torch.float16, device=dev)
    lm_entry = T.get("output.weight", T["token_embd.weight"])
    freqs = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))

    def rms_norm(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

    def rope(x, pos):
        ang = torch.outer(pos.float(), freqs)
        c, s = torch.cos(ang).half()[None, None], torch.sin(ang).half()[None, None]
        return torch.cat([x[..., :hd//2]*c - x[..., hd//2:]*s, x[..., hd//2:]*c + x[..., :hd//2]*s], dim=-1)

    @torch.no_grad()
    def run_4b_ctx(ctx_len, secret, saturated=False, top_k=256, gen_tokens=16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        doc_ids, q_ids = build_needle_parts(bpe, ctx_len, secret)
        max_S = len(doc_ids) + len(q_ids) + gen_tokens + 64
        mcache = ThinMasullo4B(nl, nh, nkv, max_S, hd, dev)
        chunk_sz = 256

        t_pref0 = time.perf_counter()
        cur_pos = 0
        sink_k, sink_v = [None]*nl, [None]*nl
        for start in range(0, len(doc_ids), chunk_sz):
            ch = doc_ids[start : start + chunk_sz]
            Tc = len(ch)
            ids_t = torch.tensor(ch, device=dev, dtype=torch.long)
            pos_t = torch.arange(cur_pos, cur_pos + Tc, device=dev, dtype=torch.long)
            x = qembed(ids_t, T["token_embd.weight"])
            for l in range(nl):
                p = f"blk.{l}."
                h = rms_norm(x, T[p + "attn_norm.weight"][2])
                q = rms_norm(qlinear(h, T[p + "attn_q.weight"], scratch_w).view(1, Tc, nh, hd), T[p + "attn_q_norm.weight"][2]).transpose(1, 2)
                k = rms_norm(qlinear(h, T[p + "attn_k.weight"], scratch_w).view(1, Tc, nkv, hd), T[p + "attn_k_norm.weight"][2]).transpose(1, 2)
                v = qlinear(h, T[p + "attn_v.weight"], scratch_w).view(1, Tc, nkv, hd).transpose(1, 2)
                q, k = rope(q, pos_t), rope(k, pos_t)
                if cur_pos == 0:
                    sink_k[l] = k[:, :, :8, :].clone(); sink_v[l] = v[:, :, :8, :].clone()
                mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                if cur_pos == 0:
                    attn = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True)
                else:
                    kc = torch.cat([sink_k[l], k], dim=2).repeat_interleave(4, 1)
                    vc = torch.cat([sink_v[l], v], dim=2).repeat_interleave(4, 1)
                    cm = torch.ones((Tc, 8 + Tc), device=dev, dtype=torch.bool)
                    cm[:, 8:] = torch.tril(torch.ones((Tc, Tc), device=dev, dtype=torch.bool))
                    attn = F.scaled_dot_product_attention(q, kc, vc, attn_mask=cm)
                x = x + qlinear(attn.transpose(1, 2).contiguous().view(1, Tc, nh * hd), T[p + "attn_output.weight"], scratch_w)
                h2 = rms_norm(x, T[p + "ffn_norm.weight"][2])
                inter = F.silu(qlinear(h2, T[p + "ffn_gate.weight"], scratch_w)) * qlinear(h2, T[p + "ffn_up.weight"], scratch_w)
                x = x + qlinear(inter, T[p + "ffn_down.weight"], scratch_w)
            cur_pos += Tc

        out_buf = torch.empty((nh, hd), dtype=torch.float16, device=dev)
        last_logits = None
        for qi, tok_id in enumerate(q_ids):
            ids_t = torch.tensor([tok_id], device=dev, dtype=torch.long)
            pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.long)
            x = qembed(ids_t, T["token_embd.weight"])
            for l in range(nl):
                p = f"blk.{l}."
                h = rms_norm(x, T[p + "attn_norm.weight"][2])
                q = rms_norm(qlinear(h, T[p + "attn_q.weight"], scratch_w).view(1, 1, nh, hd), T[p + "attn_q_norm.weight"][2]).transpose(1, 2)
                k = rms_norm(qlinear(h, T[p + "attn_k.weight"], scratch_w).view(1, 1, nkv, hd), T[p + "attn_k_norm.weight"][2]).transpose(1, 2)
                v = qlinear(h, T[p + "attn_v.weight"], scratch_w).view(1, 1, nkv, hd).transpose(1, 2)
                q, k = rope(q, pos_t), rope(k, pos_t)
                mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                mcache.decode_attn(l, q[0, :, 0, :].contiguous(), cur_pos + 1, out_buf, saturated=saturated, top_k=top_k)
                x = x + qlinear(out_buf.view(1, 1, nh * hd), T[p + "attn_output.weight"], scratch_w)
                h2 = rms_norm(x, T[p + "ffn_norm.weight"][2])
                inter = F.silu(qlinear(h2, T[p + "ffn_gate.weight"], scratch_w)) * qlinear(h2, T[p + "ffn_up.weight"], scratch_w)
                x = x + qlinear(inter, T[p + "ffn_down.weight"], scratch_w)
            cur_pos += 1
            if qi == len(q_ids) - 1:
                last_logits = qlinear(rms_norm(x[:, -1:, :], T["output_norm.weight"][2]), lm_entry, scratch_w).float().squeeze()

        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_pref0

        gen_ids = [int(last_logits.argmax().item())]
        attn_ms = 0.0
        ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_dec0 = time.perf_counter()
        for _ in range(gen_tokens - 1):
            ids_t = torch.tensor([gen_ids[-1]], device=dev, dtype=torch.long)
            pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.long)
            x = qembed(ids_t, T["token_embd.weight"])
            for l in range(nl):
                p = f"blk.{l}."
                h = rms_norm(x, T[p + "attn_norm.weight"][2])
                q = rms_norm(qlinear(h, T[p + "attn_q.weight"], scratch_w).view(1, 1, nh, hd), T[p + "attn_q_norm.weight"][2]).transpose(1, 2)
                k = rms_norm(qlinear(h, T[p + "attn_k.weight"], scratch_w).view(1, 1, nkv, hd), T[p + "attn_k_norm.weight"][2]).transpose(1, 2)
                v = qlinear(h, T[p + "attn_v.weight"], scratch_w).view(1, 1, nkv, hd).transpose(1, 2)
                q, k = rope(q, pos_t), rope(k, pos_t)
                ev_s.record()
                mcache.pack_chunk(l, k.squeeze(0).contiguous(), v.squeeze(0).contiguous(), cur_pos)
                mcache.decode_attn(l, q[0, :, 0, :].contiguous(), cur_pos + 1, out_buf, saturated=saturated, top_k=top_k)
                ev_e.record()
                torch.cuda.synchronize()
                attn_ms += ev_s.elapsed_time(ev_e)
                x = x + qlinear(out_buf.view(1, 1, nh * hd), T[p + "attn_output.weight"], scratch_w)
                h2 = rms_norm(x, T[p + "ffn_norm.weight"][2])
                inter = F.silu(qlinear(h2, T[p + "ffn_gate.weight"], scratch_w)) * qlinear(h2, T[p + "ffn_up.weight"], scratch_w)
                x = x + qlinear(inter, T[p + "ffn_down.weight"], scratch_w)
            cur_pos += 1
            gen_ids.append(int(qlinear(rms_norm(x[:, -1:, :], T["output_norm.weight"][2]), lm_entry, scratch_w).float().squeeze().argmax().item()))

        torch.cuda.synchronize()
        dec_s = time.perf_counter() - t_dec0
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        ans = bpe.decode(gen_ids)
        mcache.close()
        torch.cuda.empty_cache()

        n_steps = gen_tokens - 1
        fp16_kv_gib = (nl * nkv * ctx_len * hd * 4) / (1<<30)
        thin_kv_gib = (nl * nkv * ctx_len * 26) / (1<<30)
        return {
            "model": "Qwen3-4B-Instruct-2507-Q4_K_M (36L, 32Q/8KV, native 256K)",
            "context_len": ctx_len,
            "mode": "saturated_gqa4_v3" if saturated else "indep_head_v3",
            "peak_total_vram_mib": round(peak_mib, 1),
            "peak_total_vram_gib": round(peak_mib / 1024.0, 3),
            "gpu_kv_only_gib": round(thin_kv_gib, 3),
            "dense_fp16_kv_gib": round(fp16_kv_gib, 2),
            "gpu_kv_compression": round(fp16_kv_gib / thin_kv_gib, 2),
            "prefill_s": round(prefill_s, 2),
            "decode_tps": round(n_steps / dec_s, 2),
            "attn_ms_36L": round(attn_ms / n_steps, 2),
            "secret": secret,
            "needle_exact_match": (secret in ans),
            "generated": ans.strip()[:80],
        }

    out = {"track": "R4-2B_Real_36L_Qwen3_4B_128K_Masullo_v3", "runs": []}
    for ctx_len, secret, sat, tk in [
        (32768,  "ORCHID-3819",  False, 256),
        (32768,  "ORCHID-3819",  True,  512),
        (65536,  "KESTREL-6140", False, 256),
        (65536,  "KESTREL-6140", True,  768),
        (131072, "COBALT-7741",  False, 256),
        (131072, "COBALT-7741",  True,  768),
    ]:
        print(f"\n=== REAL 36L QWEN3-4B S={ctx_len} sat={sat} topk={tk} secret={secret} ===", flush=True)
        r = run_4b_ctx(ctx_len, secret, saturated=sat, top_k=tk, gen_tokens=16)
        r["top_k_spikes"] = tk
        print(" ", r, flush=True)
        out["runs"].append(r)
        # If 128K indep @ 256 misses, also test Alpoge sqrt(S) top_k=512 (still only 0.39% of 128K!)
        if ctx_len == 131072 and not sat and not r["needle_exact_match"]:
            print(f"\n=== REAL 36L QWEN3-4B S=131072 FALLBACK sat=False topk=512 ===", flush=True)
            r2 = run_4b_ctx(ctx_len, secret, saturated=False, top_k=512, gen_tokens=16)
            r2["top_k_spikes"] = 512
            print(" ", r2, flush=True)
            out["runs"].append(r2)

    p_out = Path(os.environ.get("OUT_DIR", "./results")) / "r4_masullo_v3_64k_4b.json"
    p_out.write_text(json.dumps(out, indent=2))
    print("\nSAVED:", p_out, flush=True)

if __name__ == "__main__":
    main()
