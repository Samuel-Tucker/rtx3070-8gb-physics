# rtx3070-8gb-physics

First-principles CUDA kernels, `llama.cpp` batching & speculative decoding benchmarks, and raw JSON measurements for breaking the conventional physics of running LLMs on an **8 GB laptop GPU** (NVIDIA GeForce RTX 3070 Laptop GPU, Ampere `sm_86`, 80 W TGP).

![Four measured wins on an 8 GB RTX 3070 Laptop GPU](assets/hero.png)

* **Full essay (Round 1):** [What a Riemann Note Taught My 8 GB Laptop GPU](https://sams.help/essays/riemann-note-8gb-laptop-gpu/)
* **X thread / article:** [@scanningwithsam](https://x.com/scanningwithsam/status/2123899077435752795)

> **Note on mathematical inspiration:** Several kernel designs below were inspired by structural moves in two **unverified** November 20, 2026 GRH proof preprints — Levent Alpöge & Claude's `alpo.ge/remark.pdf` (Rallis seesaw, Parseval FWHT, Iwasawa nested bit-planes, Bost maximal-slope inequality) and Alessandro Maria Masullo's *The Music of the Primes* (global-plus-local Shell Structure and bare-minimum normed topologies). Both mathematical claims remain **unverified** and are used here strictly as algorithmic inspiration. Every GPU claim in this repo is independently measured on hardware and backed by raw JSON in [`results/`](results/).

---

## Use Today: Copy-Paste `llama-server` Commands (RTX 3070 8 GB)

Single-token decode wastes ~98% of the Tensor Cores and hits a sharp `M=8` latency valley (`pp8` = `44.1 ms` vs `pp16` = `28.8 ms` on `Qwen3 8B Q4_K_M`).

**1. Zero extra VRAM (`+0 MiB`, `5.39 GiB` total) — general prose & code:**
```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa on -c 8192 \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-max 15
```
*Measured (`100%` char-exact greedy):* `81.96 tok/s` (`1.73×`) on technical prose, `156.25 tok/s` (`3.40×`) on code refactoring.

**2. Round 2 Stacked Winner (`+0.88 GiB` VRAM, `6.25 GiB` total) — breaks `2×` on fresh novel prose AND `6.19×` on code:**
```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf \
  -md Qwen3-0.6B-Q4_K_M.gguf -ngld 99 \
  -ngl 99 -fa on -c 4096 \
  --spec-type ngram-mod,draft-simple \
  --spec-ngram-mod-n-max 15 \
  --spec-draft-n-max 15 \
  --spec-draft-p-min 0.5
```
*Measured on RTX 3070 Laptop (`results/r2_llama_prose_v2.json`):*
* **Fresh technical prose:** `46.24 → 93.83 tok/s` (**`2.03×`**, `89.5%` acceptance — `+66%` faster than `ngram-mod` alone `56.60 tok/s`, and `+22%` faster than serial `Qwen3-4B` `76.92 tok/s` at full `8B` fidelity!)
* **Code refactoring:** `44.18 → 273.62 tok/s` (**`6.19×`**, `94.4%` acceptance)
* **Hard math reasoning:** `45.26 → 44.85 tok/s` (`0.99×` — confidence gate `--spec-draft-p-min 0.5` prevents the un-gated `24.68 tok/s` slowdown!)
* *(Note: Target model verifies every draft token losslessly; standard FP16 `M=1` vs `M=16` ULP tie-breaking can occasionally flip near-tied greedy logits compared to serial `M=1`.)*

**3. Heavy code refactoring (`+0 MiB` VRAM, `ngram-simple k=31`):**
```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf \
  -ngl 99 -fa on -c 8192 \
  --spec-type ngram-simple \
  --spec-ngram-simple-size-n 6 \
  --spec-ngram-simple-size-m 31 \
  --spec-draft-n-max 31
```
*Measured (`722/722` chars exact match, `results/verify_k31_exact.json`):* `48.84 → 414.37 tok/s` (**`8.48×`**, `94.8%` acceptance).

---

## Hardware Physics Baseline (RTX 3070 Laptop, 80 W)

Measured with [`benchmarks/hw_physics.py`](benchmarks/hw_physics.py) and [`benchmarks/exp0_cudagraph_and_80w.py`](benchmarks/exp0_cudagraph_and_80w.py):

| Metric | Measured Value | Takeaway |
| :--- | :--- | :--- |
| Architecture / SMs | Ampere `sm_86`, 40 SMs | 100 KiB SMEM/SM, 4.00 MiB L2, 8,192 MiB GDDR6 (7.66 GiB PyTorch) |
| GDDR6 copy bandwidth | `358.7 GB/s` | Real decode ceiling |
| L2 cache copy (1.5 MiB) | `523.3 GB/s` | Must stay inside 4.00 MiB L2 (`d=1024` = 2 MiB; `3.76×` cliff at `d=1536` = 4.5 MiB) |
| PCIe Gen3 x16 pinned H2D | `6.44 GB/s` | **55.7× slower** than GDDR6 — never stream dense weights/KV over PCIe; *do* fetch `16` exact KV spikes (`8 KiB`) |
| Sustained 80 W FP16 GEMM (`4096×4096`) | `M=1`: `0.1218 ms` (0.28 TFLOPS)<br>`M=32`: `0.1190 ms` (`0.98×`, 9.02 TFLOPS)<br>`M=64`: `0.1365 ms` (`1.12×`, 15.73 TFLOPS) | Batching up to 32–64 tokens is nearly free on an 80 W card |
| CUDA Graph launch overhead | `d=256`: `14.87 → 2.68 µs` (**5.55×**)<br>`d=1024`: `15.80 → 6.37 µs` (**2.48×**) | Small sub-20 µs kernels must run inside CUDA Graphs |

---

## Round 3 Live End-to-End Proof (Win #8 — Real 28-Layer 64K Needle-in-a-Haystack on 8 GB VRAM!)

We compiled Masullo-Shell Sign-KV v2.1 into a live CUDA shared library ([`kernels/masullo_kv_lib.cu`](kernels/masullo_kv_lib.cu)) and wired it directly into **all 28 real layers of frozen `Qwen3-0.6B`** (`16` Q heads, `8` KV heads, `head_dim=128`) on the clean 8 GB RTX 3070 Laptop — testing real secret passcode retrieval at 50% depth across **8K, 32K, and 64K context** ([`benchmarks/r3_e2e_64k_needle.py`](benchmarks/r3_e2e_64k_needle.py), [`results/r3_e2e_64k_needle.json`](results/r3_e2e_64k_needle.json)):

| Context (`S`) | Mode | Peak GPU VRAM | 28-Layer Attn / tok | End-to-End Decode | Secret Passcode Retrieved? |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **8,192 (8K)** | Dense FP16 SDPA | `2,442 MiB` | `16.92 ms` | `21.69 tok/s` | **YES** (`FALCON-9472`) |
| **8,192 (8K)** | **Masullo-Shell v2.1** | **`1,627 MiB`** | **`15.77 ms`** | **`22.15 tok/s`** | **YES** (`FALCON-9472`) |
| **32,768 (32K)** | Dense FP16 SDPA | `5,418 MiB` | `72.34 ms` | `9.66 tok/s` | **YES** (`ORCHID-3819`) |
| **32,768 (32K)** | **Masullo-Shell v2.1** | **`2,038 MiB` (`2.66×` less!)** | **`21.02 ms` (`3.44×` faster!)** | **`19.03 tok/s` (`1.97×` faster!)** | **YES** (`ORCHID-3819`) |
| **65,536 (64K)** | Dense FP16 SDPA | **CUDA OOM** (`7.01 GiB` KV alone) | — | — | **OOM** |
| **65,536 (64K)** | **Masullo-Shell v2.1** | **`2,570 MiB` (`2.51 GiB` total!)** | **`27.71 ms`** | **`17.10 tok/s` (`1.77×` faster than Dense @ 32K!)** | **YES** (`KESTREL-6140`) |

### Two Real-LLM CUDA Physics Lessons Discovered During Build A:
1. **Exclude BOS/Sink Norms from `gmax` + Use a 64-Bin SMEM Adaptive Histogram:** On real `Qwen3` models, token 0 (`<|im_start|>`) has a `5–10×` larger key norm than content tokens, drowning out fixed-margin thresholds. Computing a 64-bin histogram strictly over content tokens guarantees global top-K needle selection with zero buffer overflow.
2. **The Softmax + Low-Bit Key Law:** Never feed noisy 2-bit `Q·K` dot products into `exp()` on real RoPE LLMs where logits reach `25–40` (a 15% error causes `exp(±5) ≈ 150×` softmax explosion in early layers `L01–L05`). Use 1-bit `__popc` **strictly for screening**, then score all shortlisted ~256 survivors with **exact FP16 `Q·K` over zero-copy pinned host RAM**!

---

## Round 2 Moonshot Breakthroughs (Wins #5–#7)

| # | Breakthrough | Measured Result (Clean RTX 3070) | Honest Caveat & Scope | Source |
| :--- | :--- | :--- | :--- | :--- |
| **5** | **Masullo-Shell Sign-KV v2.1 (`128K` context in `2.74 GiB` + Zero-Copy PCIe Spike Refine)** | Solves Round 1's `0.5842` cosine caveat via 3 shells: (0) 64-token page centroids `(μ_K, μ_V, var)` integrating all background mass, (1) 1-bit FWHT `__popc` screen + 2-bit Iwasawa K + 2-bit packed V in VRAM, (2) zero-copy pinned-host (`cudaHostAllocMapped`) exact FP16 K+V fetch ONLY for the `~16–17` hottest spikes (`0.013%–0.024%` of context!).<br>• **64K (`M=1`):** **`591.8 µs`** vs `6,840.9 µs` GPU (**`11.56×`** vs GPU, **`70.6×`** vs PCIe spill), **`1.37 GiB`** GPU VRAM (vs `9.00 GiB`), **`mean_cos = 1.0000`**, **`min_cos = 0.9999`** (incl. planted needles!), `1.00%` rel L2.<br>• **128K (`M=1`):** **`947.7 µs`** vs `13,125.8 µs` GPU (**`13.85×`**), **`2.74 GiB`** GPU VRAM (vs `18.00 GiB` — fits 8B + 128K inside 8 GB!), **`min_cos = 1.0000`**, `0.69%` rel L2. At `M=16/32`: **`7.38× / 6.59×`**, `min_cos ≥ 0.9999`.<br>• **Real 28-layer frozen `Qwen3-0.6B` RoPE KV (`442` tokens, `PPL=13.057`):** Top-16 exact spikes + **FWHT 2-bit V** + centroids = **`28L_cos = 0.9840`, `PPL = 13.068 (+0.08%)`**; with **4-bit V** = **`28L_cos = 0.9862`, `PPL = 13.054 (-0.03%)`**! | CUDA microbenchmark uses synthetic multi-cluster + 16-needle KV with real zero-copy PCIe spike reads; real-model 28-layer validation (`442` tokens) confirmed `+0.08%` PPL with FWHT 2-bit V (`-0.03%` with 4-bit V), while raw un-rotated 2-bit min-max V without top-16 exact spikes degraded real Qwen3 PPL by `+71.9%`. | [`kernels/sign_kv_v2_exact.cu`](kernels/sign_kv_v2_exact.cu)<br>[`kernels/sign_kv_v2.cu`](kernels/sign_kv_v2.cu)<br>[`benchmarks/r2_real_kv_v21.py`](benchmarks/r2_real_kv_v21.py)<br>[`results/r2_sign_kv_v21_exact.json`](results/r2_sign_kv_v21_exact.json)<br>[`results/r2_real_kv_v21.json`](results/r2_real_kv_v21.json) |
| **6** | **Split-K 40-SM Column-Major Sparse `W_down` CUDA Kernel (Real Wall-Clock FFN Speedup)** | Turns Round 1's PyTorch quality simulation (Win #4) into real bare-metal GDDR6 speed by fixing a 4-block occupancy trap (`0.30×` → `1.94×`!) with Split-K saturating all 40 SMs.<br>On `Qwen3-8B` shape (`4096×12288`, cold L2 flush):<br>• Dense `cuBLAS FP16` `W_down`: `289.79 µs`<br>• **50% skip (zero training!):** **`149.28 µs` (`1.94×` vs cuBLAS FP16)** (`-0.56%` R1 PPL / `+1.51%` sealed BPE PPL)<br>• **60% skip (zero training!):** **`121.73 µs` (`2.38×` vs cuBLAS FP16)** (`+0.90%` R1 PPL / `+7.18%` sealed BPE PPL)<br>• **70% / 75% skip:** **`3.07×` / `3.58×`** (and Rank-64 screened full INT4 FFN hits `1.51×–2.12×` vs dense INT4 / `5.22×–7.32×` vs FP16 if a trained router provides the mask). | `W_down` skipping works zero-training today (`1.94×` at 50% skip). Pre-screening `gate+up` *before* reading them failed zero-training quality (Kill #6) and requires a trained MLP router (DejaVu/ProSparse style). | [`kernels/coalesced_ffn_v2.cu`](kernels/coalesced_ffn_v2.cu)<br>[`results/r2_coalesced_ffn_v2.json`](results/r2_coalesced_ffn_v2.json)<br>[`results/r2_moonshots_v2.json`](results/r2_moonshots_v2.json) |
| **7** | **Stacked `ngram-mod,draft-simple` (`522 MB` `Qwen3-0.6B` + `p_min=0.5`) on Real `Qwen3-8B`** | Breaks Round 1's fresh-prose ceiling on real `Qwen3-8B Q4_K_M` (`6.25 GiB` total VRAM):<br>• **Fresh technical prose:** `46.24` (serial) → `56.60` (`ngram-mod`, `1.22×`) → **`93.83 tok/s` (`2.03×` stacked, `89.5%` acceptance)** — beating serial `Qwen3-4B` (`76.92 tok/s`) by `+22%`!<br>• **Code refactoring:** `44.18 → 273.62 tok/s` (**`6.19×`**, `94.4%` acceptance).<br>• **Hard math reasoning:** `45.26 → 44.85 tok/s` (`0.99×`, zero slowdown thanks to `p_min=0.5`). | Target verifies all draft tokens losslessly; FP16 batch-size ULP tie-breaking can differ from serial `M=1` on near-tied greedy tokens. Does not speed up `Qwen3-4B` (Kill #8). | [`benchmarks/r2_llama_prose_moonshots.py`](benchmarks/r2_llama_prose_moonshots.py)<br>[`results/r2_llama_prose_v2.json`](results/r2_llama_prose_v2.json) |

### Round 2 Architectural Law (`r2_hsal_v3.json`)
* **`M=1..4` vs `M=16..32` Hardware Split:** HSAL L2-LUT crushes `cuBLAS FP16` by **`4.70×` at `M=1`** (and fusing `3.125%` Bost-slope FP16 outlier columns adds only `+1.02 µs` / `+2.5%`), **but at `M=16..32`**, `cuBLAS` Tensor Cores stay flat at `107.5 µs` while scalar LUT (`1,280 µs`) and scalar INT4 (`795.6 µs`) scale linearly (`7×–12×` slower). **Rule:** use low-bit L2-LUT for `M=1..4` decode/drafting, and Tensor Core `mma.sync` dequant-to-FP16 for `M=16..32` verification.

---

## Round 1 Measured Wins (Wins #1–#4)

| # | Finding | Measured Result | Honest Caveat & Scope | Source |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **`llama.cpp` `M=8` valley & zero-VRAM n-gram speculation** | `Qwen3 8B Q4_K_M`: `pp8` = `44.1 ms` vs `pp16` = `28.8 ms` (-35%) & `pp32` = `34.4 ms` (-22%).<br>`ngram-simple k=31` code refactor: `48.84 → 414.37 tok/s` (**8.48×**).<br>`ngram-mod k=15` prose: `47.46 → 81.96 tok/s` (**1.73×**).<br>32 concurrent streams (`Qwen3 4B`): `76.9 → 880.6` aggregate tok/s (**11.45×**). | `414.37 tok/s` is specifically on code refactoring; all speculative outputs verified 100% char-exact at greedy temp 0. | [`results/track1b_results.json`](results/track1b_results.json)<br>[`results/track5_results.json`](results/track5_results.json)<br>[`results/verify_k31_exact.json`](results/verify_k31_exact.json) |
| **2** | **HSAL 1.125 bpw Rallis-Seesaw L2-LUT CUDA kernel** | `4096×4096, M=1` (CUDA Graph, incl. register FWHT64 + 128 KiB L2 LUT build): **`20.48 µs` warm / `22.53 µs` cold** vs `cuBLAS FP16` (`96.26 / 103.42 µs`, **4.70× / 4.59×**), `packed_int4_v2` (`32.77 / 36.86 µs`, **1.60×–1.64×**), and direct 1.125-bit register unpacking (`46.08 / 47.10 µs`, **2.25× / 2.09×**). Payload: `2.36 MB` vs `33.55 MB` FP16 (`1.97 GB` / `1.83 GiB` for 14B weights). | **Synthetic-weight microbenchmark** (`M=1`). Post-hoc rounding to 1.125–2.25 bpw failed on frozen weights without QAT (see Kill #1). | [`kernels/hsal_bench_v2.cu`](kernels/hsal_bench_v2.cu)<br>[`kernels/reg_sign_bench.cu`](kernels/reg_sign_bench.cu)<br>[`results/hsal_bench_v2.json`](results/hsal_bench_v2.json)<br>[`results/reg_sign_bench.json`](results/reg_sign_bench.json) |
| **3** | **1-bit Parseval FWHT Sign-KV v1** | `S=65,536, M=1`: **`424.7 µs`** (`15.43×` vs GPU, `98.4×` vs PCIe). Superseded by **Win #5 (Sign-KV v2.1)** which fixes the `0.5842` cosine caveat up to **`1.0000`**! | Aggressive `4.5`-margin gate kept `16/65,536` keys (`out_cos = 0.5842`). Fixed completely in Win #5. | [`kernels/sign_kv_bench.cu`](kernels/sign_kv_bench.cu)<br>[`results/track3_signkv.json`](results/track3_signkv.json) |
| **4** | **Zero-training SwiGLU FFN column skipping (quality)** | Frozen `Qwen3-0.6B` FP16 (383 targets): masking smallest **50%** of FFN activations improved PPL (`9.034 → 8.984`, **-0.56%**); **60%** cost **+0.90%** (`9.116`); cliff at **70%** (`+9.48%`). | Quality simulation only — real bare-metal wall-clock speedup (`1.94×` at 50% / `2.38×` at 60%) now proven in **Win #6**! | [`benchmarks/track5_opus_fable.py`](benchmarks/track5_opus_fable.py)<br>[`results/track5_results.json`](results/track5_results.json) |

---

## Nine Honest Kills (Published So Nobody Retraces Dead Ends)

1. **Post-hoc 1.125–2.5 bpw rounding on frozen `Qwen3-0.6B` (R1 + R2):** Naive 1.125–2.25 bpw collapsed NLL to `11.64–16.11`. Adding two-sided Hadamard (`H_N W H_K`) + 3–5% Bost-slope FP16 outlier columns slashed 2-bit NLL from `14.32 → 6.58` (`-7.74` NLL!), and `3-bit + 3% outliers` (`3.63 bpw`) reached `NLL = 5.28` (`top2 = 76.2%`), **but** sub-3-bit post-hoc still misses a `<+10% PPL` bar without QAT/GPTQ ([`results/track2_quality.json`](results/track2_quality.json), [`results/r2_moonshots_v2.json`](results/r2_moonshots_v2.json)).
2. **Post-hoc 2D-FFT / Monarch-FWHT / Sign+SVD on frozen weights (R1):** All scored `0.37–0.91` cosine (`<0.95` bar) and ran slower than dense in PyTorch ([`results/track4_spectral.json`](results/track4_spectral.json)).
3. **Naive `Qwen3-1.7B` separate draft model on prose (R1):** Added `+1.62 GiB` VRAM and slowed `Qwen3-8B` prose (`47.5 → 26.9–37.3 tok/s`). Replaced by confidence-gated `522 MB` `Qwen3-0.6B` stacked with `ngram-mod` in Win #7!
4. **Pure unconditioned Jacobi parallel decoding (`W=8, 16`) on frozen prose (R2):** Required `61` sweeps for `64` tokens (`1.049 tok/sweep`, exact greedy match) on frozen `Qwen3-0.6B` ([`results/r2_frozen_moonshots.json`](results/r2_frozen_moonshots.json)).
5. **Zero-training Bost-slope Layer-Skip self-drafting (R2):** Skipping `6–14` of `28` layers on frozen `Qwen3-0.6B` dropped top-1 agreement to `0.0%–19.1%` (`1.00–1.31` tree16 acceptance) ([`results/r2_frozen_moonshots.json`](results/r2_frozen_moonshots.json)).
6. **Zero-training `gate+up` FFN pre-screening (R2):** Both 1-bit sign (`+1,189%` PPL) and activation-weighted Rank-64/128/256 SVD (`+25%` to `+260%` PPL at 40–50% skip) failed quality on frozen `Qwen3-0.6B` because `SiLU(g)*u` is sharp and full-rank — only `W_down` skipping works zero-training ([`results/r2_frozen_moonshots.json`](results/r2_frozen_moonshots.json), [`results/r2_moonshots_v2.json`](results/r2_moonshots_v2.json)).
7. **Synchronous CPU-Ghost `0.6B` draft in `llama.cpp` (R2):** Running `Qwen3-0.6B` on the 8-core i9 (`--spec-draft-device none`, `+0 MiB` VRAM) slowed `Qwen3-8B` prose to `25.57 tok/s` (`0.55×`) because `llama.cpp` stalls GPU verification during serial CPU drafting ([`results/r2_llama_prose_v2.json`](results/r2_llama_prose_v2.json)).
8. **`Qwen3-0.6B` drafting for `Qwen3-4B` (R2):** Slowed `Qwen3-4B` prose (`76.92 → 58.46 tok/s`, `0.76×`) because the `4B` target is already fast enough that a `0.6B` drafter lacks sufficient speed ratio ([`results/r2_llama_prose_v2.json`](results/r2_llama_prose_v2.json)).
9. **Small Trained 2-Layer ReLU MLP Router (`R=128, 256`) for SwiGLU `gate+up` Pre-Screening (R3):** Trained across all 28 layers of `Qwen3-0.6B` on disjoint prose (`1,536` tokens), still degraded sealed PPL by `+54.7%–72.5%` at 40% skip and `+125%–128%` at 50% skip (whereas `oracle_down_only` was `-0.03%` at 40% and `+0.52%` at 50%!). Unlike ReLU models (DejaVu), frozen SwiGLU lacks hard zeros and requires ReLU-ification or QAT to pre-skip `gate+up` ([`benchmarks/r3_dejavu_ffn_router.py`](benchmarks/r3_dejavu_ffn_router.py), [`results/r3_dejavu_ffn_router.json`](results/r3_dejavu_ffn_router.json)).

---

## Building & Running the CUDA Kernels

Tested on Linux with CUDA 12.x / 13.x on Ampere `sm_86` (change `-arch=sm_86` for your GPU):

```bash
# Round 3 Live End-to-End 64K Masullo-Shell Library + Real 28-Layer Qwen3-0.6B Needle Test
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math -Xcompiler -fPIC -shared kernels/masullo_kv_lib.cu -o libmasullo_kv.so
MASULLO_LIB=./libmasullo_kv.so QWEN3_06B_DIR=/path/to/Qwen3-0.6B python3 benchmarks/r3_e2e_64k_needle.py

# Round 2 Moonshot Kernels
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/sign_kv_v2_exact.cu -o sign_kv_v2_exact && ./sign_kv_v2_exact
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/sign_kv_v2.cu -o sign_kv_v2 && ./sign_kv_v2
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/coalesced_ffn_v2.cu -lcublas -o coalesced_ffn_v2 && ./coalesced_ffn_v2
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/hsal_v3_bench.cu -lcublas -o hsal_v3_bench && ./hsal_v3_bench

# Round 1 Kernels
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/hsal_bench_v2.cu -lcublas -o hsal_bench_v2 && ./hsal_bench_v2
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/reg_sign_bench.cu -o reg_sign_bench && ./reg_sign_bench
nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math kernels/sign_kv_bench.cu -o sign_kv_bench && ./sign_kv_bench
```

## Running the Python Benchmarks

```bash
# Hardware ceilings & CUDA Graphs / 80 W sweep
python3 benchmarks/hw_physics.py
python3 benchmarks/exp0_cudagraph_and_80w.py

# Round 2 Real-Model Qwen3-0.6B Suites (set QWEN3_06B_DIR to local HuggingFace snapshot)
QWEN3_06B_DIR=/path/to/Qwen3-0.6B python3 benchmarks/r2_real_kv_v21.py
QWEN3_06B_DIR=/path/to/Qwen3-0.6B python3 benchmarks/r2_moonshots_v2.py
QWEN3_06B_DIR=/path/to/Qwen3-0.6B python3 benchmarks/r2_frozen_moonshots.py

# Round 2 Stacked ngram-mod + 0.6B draft on Qwen3-8B / 4B
LLAMA_SERVER_BIN=/path/to/llama-server \
  QWEN3_8B_GGUF=/path/to/Qwen3-8B-Q4_K_M.gguf \
  QWEN3_4B_GGUF=/path/to/Qwen3-4B-Q4_K_M.gguf \
  QWEN3_06B_GGUF=/path/to/Qwen3-0.6B-Q4_K_M.gguf \
  python3 benchmarks/r2_llama_prose_moonshots.py
```

---

## License

MIT — see [`LICENSE`](LICENSE). Fork it, benchmark it on your own GPU, or turn Win #5 (Masullo-Shell Sign-KV v2.1), Win #6 (Split-K Sparse `W_down`), or Win #2 (HSAL + QAT) into production inference engines.
