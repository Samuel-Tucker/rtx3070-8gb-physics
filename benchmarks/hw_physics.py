import time, torch, json

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
res = {
    "gpu": props.name,
    "sm_count": props.multi_processor_count,
    "vram_gib": round(props.total_memory / 1024**3, 3),
    "l2_mib": round(props.L2_cache_size / 1024**2, 2),
    "shmem_per_sm_kib": props.shared_memory_per_multiprocessor // 1024,
    "total_shmem_mib": round(props.multi_processor_count * props.shared_memory_per_multiprocessor / 1024**2, 2),
}

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

# 1. Memory bandwidth sweep (copy: read + write = 2x bytes)
bw = {}
for kb in [128, 512, 1536, 3072, 6144, 16384, 65536, 262144]:
    n = (kb * 1024) // 4
    a = torch.empty(n, dtype=torch.float32, device=dev)
    b = torch.empty(n, dtype=torch.float32, device=dev)
    ms = bench_ms(lambda: b.copy_(a), warmup=20, iters=100 if kb <= 16384 else 40)
    gbps = (2 * kb * 1024) / (ms * 1e-3) / 1e9
    bw[f"{kb}KB"] = round(gbps, 1)
res["copy_bw_gbps"] = bw

# 2. Compute & GEMV vs GEMM sweep (4096 x 4096 FP16)
W = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
gemm = {}
for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
    X = torch.randn(m, 4096, dtype=torch.float16, device=dev)
    ms = bench_ms(lambda: torch.matmul(X, W), warmup=20, iters=80 if m <= 128 else 30)
    flops = 2 * m * 4096 * 4096
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_read = 2 * (4096 * 4096 + m * 4096 + m * 4096)
    eff_gbps = bytes_read / (ms * 1e-3) / 1e9
    gemm[f"M={m}"] = {"ms": round(ms, 4), "tflops": round(tflops, 2), "eff_gbps": round(eff_gbps, 1)}
res["fp16_4096x4096"] = gemm

# 3. Small matrix inside L2 (1024x1024 FP16 = 2 MB) vs 512x512 (0.5 MB) vs 4096x4096 (32 MB)
l2_gemv = {}
for d in [256, 512, 1024, 1536, 2048, 4096]:
    Wm = torch.randn(d, d, dtype=torch.float16, device=dev)
    Xm = torch.randn(1, d, dtype=torch.float16, device=dev)
    ms = bench_ms(lambda: torch.matmul(Xm, Wm), warmup=30, iters=200)
    gbps = (2 * d * d) / (ms * 1e-3) / 1e9
    l2_gemv[f"d={d}({d*d*2/1024**2:.2f}MB)"] = {"us": round(ms*1000, 1), "eff_gbps": round(gbps, 1)}
res["fp16_gemv_by_size"] = l2_gemv

# 4. FFT throughput (cuFFT FP16/FP32)
x_fft = torch.randn(1, 4096, dtype=torch.float32, device=dev)
ms_fft1 = bench_ms(lambda: torch.fft.rfft(x_fft), warmup=30, iters=200)
x_fft_b = torch.randn(64, 4096, dtype=torch.float32, device=dev)
ms_fft64 = bench_ms(lambda: torch.fft.rfft(x_fft_b), warmup=30, iters=200)
res["rfft_4096"] = {"B1_us": round(ms_fft1*1000, 1), "B64_us": round(ms_fft64*1000, 1)}

# 5. PCIe pinned H2D / D2H (256 MB)
cpu_pin = torch.empty(64*1024*1024, dtype=torch.float32, pin_memory=True)
gpu_buf = torch.empty(64*1024*1024, dtype=torch.float32, device=dev)
ms_h2d = bench_ms(lambda: gpu_buf.copy_(cpu_pin, non_blocking=True), warmup=5, iters=20)
ms_d2h = bench_ms(lambda: cpu_pin.copy_(gpu_buf, non_blocking=True), warmup=5, iters=20)
res["pcie_pinned_gbps"] = {"H2D": round(256*1024**2/(ms_h2d*1e-3)/1e9, 2), "D2H": round(256*1024**2/(ms_d2h*1e-3)/1e9, 2)}

print(json.dumps(res, indent=2))
