import torch, json, subprocess, threading, time

dev = torch.device("cuda:0")
out = {}

# 1. Eager vs CUDA Graph small-matrix GEMV floor (Codex Sol check!)
cg = {}
for d in [256, 512, 1024, 1536, 2048, 4096]:
    W = torch.randn(d, d, dtype=torch.float16, device=dev)
    X = torch.randn(1, d, dtype=torch.float16, device=dev)
    # Eager (500 iters inside events)
    for _ in range(50): Y = torch.matmul(X, W)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(500): Y = torch.matmul(X, W)
    e.record(); torch.cuda.synchronize()
    eager_us = s.elapsed_time(e) * 1000 / 500

    # CUDA Graph (replay 100-op chain inside single graph to eliminate host dispatch!)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            Y = torch.matmul(X, W)
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    s.record()
    for _ in range(20): g.replay()
    e.record(); torch.cuda.synchronize()
    graph_us = s.elapsed_time(e) * 1000 / (20 * 100)
    gbps_graph = (2 * d * d) / (graph_us * 1e-6) / 1e9
    cg[f"d={d}({d*d*2/1024**2:.2f}MB)"] = {
        "eager_us": round(eager_us, 2),
        "graph_us": round(graph_us, 2),
        "speedup": round(eager_us / graph_us, 2),
        "graph_eff_gbps": round(gbps_graph, 1),
    }
out["eager_vs_cudagraph_gemv"] = cg

# 2. Sustained 80W thermal/power knee across M=1,4,8,16,32,64,128 (3 seconds sustained each!)
W = torch.randn(4096, 4096, dtype=torch.float16, device=dev)
knee = {}
for m in [1, 4, 8, 16, 32, 64, 128]:
    X = torch.randn(m, 4096, dtype=torch.float16, device=dev)
    # Burn for 2.5s while sampling power & clock
    samples = []
    stop = False
    def poller():
        while not stop:
            r = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            p, c, t = [float(v.strip()) for v in r.split(",")]
            samples.append((p, c, t))
            time.sleep(0.15)
    th = threading.Thread(target=poller, daemon=True)
    th.start()
    # warmup 0.5s
    t0 = time.time()
    while time.time() - t0 < 0.5:
        _ = torch.matmul(X, W)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    iters = 4000 if m <= 64 else 1500
    s.record()
    for _ in range(iters):
        _ = torch.matmul(X, W)
    e.record(); torch.cuda.synchronize()
    stop = True; th.join()
    ms = s.elapsed_time(e) / iters
    avg_w = sum(x[0] for x in samples[-8:]) / max(1, len(samples[-8:]))
    avg_clk = sum(x[1] for x in samples[-8:]) / max(1, len(samples[-8:]))
    max_t = max(x[2] for x in samples)
    tflops = (2 * m * 4096 * 4096) / (ms * 1e-3) / 1e12
    knee[f"M={m}"] = {
        "ms": round(ms, 4),
        "ratio_vs_M1": None,
        "tflops": round(tflops, 2),
        "watts": round(avg_w, 1),
        "sm_mhz": round(avg_clk, 0),
        "temp_c": max_t,
    }
m1_ms = knee["M=1"]["ms"]
for k in knee:
    knee[k]["ratio_vs_M1"] = round(knee[k]["ms"] / m1_ms, 2)
out["sustained_80w_knee"] = knee
print(json.dumps(out, indent=2))
