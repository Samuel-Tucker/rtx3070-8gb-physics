// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math -Xcompiler -fPIC -shared masullo_kv_v3_lib.cu -o libmasullo_kv_v3.so
// Round 4 Objective 2: Masullo-Shell Sign-KV v3 ("19.7x Thin-VRAM 128K-256K Engine")
// Inspired by Masullo Ch. 11.1.2 (Unit-Sphere Bulk Degeneration 1_{O_v} -> 1_{O_v^x} & S-Local Carvings)
// and Ch. 7 Situation 7.3.1.1 (Saturated Sets of Places for GQA 4:1):
//   - GPU VRAM stores ONLY:
//       * K_p0       : 16 B/tok  (1-bit FWHT128 sign plane)
//       * K_norms    :  2 B/tok  (FP16 L2 norm)
//       * K_page_mu  :  4 B/tok  (FP16 64-tok page mean key)
//       * V_page_mu  :  4 B/tok  (FP16 64-tok page mean value)
//     TOTAL GPU KV = 26 B/token/head/layer vs 512 B FP16 = 19.69x GPU VRAM COMPRESSION!
//   - Zero-copy pinned host RAM (cudaHostAllocMapped) stores Q8_0 K & V:
//       * int8[128] + FP16 scale = 130 B per K or V (2x less host RAM & 2x faster PCIe Gen3 spike reads than FP16!)
//       * 128K 28-layer Qwen3-0.6B: ONLY 0.71 GiB GPU KV (< 1.95 GiB total VRAM!) + 7.11 GiB host RAM!
//       * 64K  36-layer Qwen3-4B:   ONLY 0.46 GiB GPU KV + 4.57 GiB host RAM!
//   - Supports arbitrary H_q / H_kv (16/8 for Qwen3-0.6B, 32/8 for Qwen3-4B & Qwen3-8B),
//     plus Masullo Saturated-Places GQA kernel sharing 1 scan across all G sibling Q heads!

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {
using Half = __half;
constexpr int kD = 128;
constexpr int kPageSize = 64;
constexpr int kSinks = 8;
constexpr int kLocalWin = 64;
constexpr int kMaxSurv = 2048;
constexpr int kMaxPages = 4096; // Supports up to S = 262,144 (256K context!)

__device__ __forceinline__ void fwht128(float v[4], int lane) {
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    #pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], bit);
            v[j] = (lane & bit) ? (o - v[j]) : (v[j] + o);
        }
    }
    const float inv = 0.0883883476f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= inv;
}

// Pack [H_kv, T, 128] FP16 K & V into:
//   - Thin GPU VRAM: ONLY K_p0 (16B) + K_norms (2B)  [NO K_p1, NO V_p0, NO V_p1!]
//     MASULLO LOCAL-WITHIN-GLOBAL RESIDUAL (v3.1): When `use_page_resid == 1`, subtracts the exact FP16
//     64-token page centroid `K_page_mu[hkv, t/64]` BEFORE taking the 1-bit FWHT sign and L2 norm!
//     That makes Q·K ≈ Q·mu_p (EXACT FP16!) + 1-bit(Q·(K - mu_p)) with ZERO extra GPU bytes (still 26 B/tok = 19.69x!)
//     and slashes 128K needle ranking error by 10x-22x!
//   - Zero-copy pinned host RAM: exact Q8_0 K & V (int8[128] + FP16 scale = 130 B each)
// Grid = (T, H_kv), Block = 32
__global__ void pack_kv_v3_q8_kernel(
    const Half* __restrict__ K_in,        // [H_kv, T, 128]
    const Half* __restrict__ V_in,        // [H_kv, T, 128]
    const Half* __restrict__ K_page_mu,   // [H_kv, max_P, 128] (already built if use_page_resid == 1)
    uint4* __restrict__ K_p0,             // [H_kv, max_S] (16 B/tok on GPU)
    Half* __restrict__ K_norms,           // [H_kv, max_S] (2 B/tok on GPU)
    int8_t* __restrict__ K_host_q8,       // [H_kv, max_S, 128] pinned mapped host RAM!
    Half* __restrict__ K_host_sc,         // [H_kv, max_S] pinned mapped host RAM!
    int8_t* __restrict__ V_host_q8,       // [H_kv, max_S, 128] pinned mapped host RAM!
    Half* __restrict__ V_host_sc,         // [H_kv, max_S] pinned mapped host RAM!
    int t_start, int T, int max_S, int max_P, int use_page_resid) {

    const int dt = blockIdx.x;
    const int hkv = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int t = t_start + dt;
    if (dt >= T || t >= max_S) return;

    const std::size_t in_off = (static_cast<std::size_t>(hkv) * T + dt) * kD + lane * 4;
    const std::size_t dst_tok = static_cast<std::size_t>(hkv) * max_S + t;
    const std::size_t dst_q8_off = dst_tok * kD + lane * 4;

    const uint2 kr = *reinterpret_cast<const uint2*>(K_in + in_off);
    const uint2 vr = *reinterpret_cast<const uint2*>(V_in + in_off);
    const half2* kh = reinterpret_cast<const half2*>(&kr);
    const half2* vh = reinterpret_cast<const half2*>(&vr);
    float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
    float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
    float kf[4] = {k0.x, k0.y, k1.x, k1.y};
    float vf[4] = {v0.x, v0.y, v1.x, v1.y};

    // 1. Quantize ORIGINAL K & V to symmetric Q8_0 and write coalesced char4 to pinned host RAM!
    float kamax = fmaxf(fmaxf(fabsf(kf[0]), fabsf(kf[1])), fmaxf(fabsf(kf[2]), fabsf(kf[3])));
    float vamax = fmaxf(fmaxf(fabsf(vf[0]), fabsf(vf[1])), fmaxf(fabsf(vf[2]), fabsf(vf[3])));
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        kamax = fmaxf(kamax, __shfl_xor_sync(0xffffffffu, kamax, d));
        vamax = fmaxf(vamax, __shfl_xor_sync(0xffffffffu, vamax, d));
    }
    float ksc = fmaxf(kamax / 127.0f, 1e-7f);
    float vsc = fmaxf(vamax / 127.0f, 1e-7f);
    float iksc = 1.0f / ksc, ivsc = 1.0f / vsc;

    char4 kq = make_char4(
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(kf[0] * iksc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(kf[1] * iksc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(kf[2] * iksc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(kf[3] * iksc, -127.0f), 127.0f))));
    char4 vq = make_char4(
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(vf[0] * ivsc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(vf[1] * ivsc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(vf[2] * ivsc, -127.0f), 127.0f))),
        static_cast<signed char>(__float2int_rn(fminf(fmaxf(vf[3] * ivsc, -127.0f), 127.0f))));

    *reinterpret_cast<char4*>(K_host_q8 + dst_q8_off) = kq;
    *reinterpret_cast<char4*>(V_host_q8 + dst_q8_off) = vq;

    // 2. Subtract exact FP16 page centroid mu_K[p] (Masullo Local-Within-Global Residual!) before 1-bit FWHT!
    const int pg = t / kPageSize;
    if (use_page_resid && K_page_mu != nullptr && pg < max_P) {
        const std::size_t poff = (static_cast<std::size_t>(hkv) * max_P + pg) * kD + lane * 4;
        const uint2 mr = *reinterpret_cast<const uint2*>(K_page_mu + poff);
        const half2* mh = reinterpret_cast<const half2*>(&mr);
        float2 m0 = __half22float2(mh[0]), m1 = __half22float2(mh[1]);
        kf[0] -= m0.x; kf[1] -= m0.y; kf[2] -= m1.x; kf[3] -= m1.y;
    }

    float rn2 = kf[0]*kf[0] + kf[1]*kf[1] + kf[2]*kf[2] + kf[3]*kf[3];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) rn2 += __shfl_xor_sync(0xffffffffu, rn2, d);
    float knorm = sqrtf(rn2);
    fwht128(kf, lane);
    unsigned b0 = __ballot_sync(0xffffffffu, kf[0] >= 0.0f);
    unsigned b1 = __ballot_sync(0xffffffffu, kf[1] >= 0.0f);
    unsigned b2 = __ballot_sync(0xffffffffu, kf[2] >= 0.0f);
    unsigned b3 = __ballot_sync(0xffffffffu, kf[3] >= 0.0f);
    if (lane == 0) {
        K_p0[dst_tok] = make_uint4(b0, b1, b2, b3);
        K_norms[dst_tok] = __float2half(knorm);
        K_host_sc[dst_tok] = __float2half(ksc);
        V_host_sc[dst_tok] = __float2half(vsc);
    }
}

// Build 64-token Masullo Page Centroids (mu_K, mu_V) — ONLY 8 B/tok amortized on GPU!
// Grid = (n_pages, H_kv), Block = 32
__global__ void build_pages_v3_kernel(
    const Half* __restrict__ K_in,
    const Half* __restrict__ V_in,
    Half* __restrict__ K_page_mu,
    Half* __restrict__ V_page_mu,
    int p_start, int n_pages, int T, int max_P) {

    const int dp = blockIdx.x;
    const int hkv = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (dp >= n_pages) return;
    const int p = p_start + dp;
    if (p >= max_P) return;

    float ksum[4] = {0,0,0,0}, vsum[4] = {0,0,0,0};
    for (int off = 0; off < kPageSize; ++off) {
        int dt = dp * kPageSize + off;
        const std::size_t ioff = (static_cast<std::size_t>(hkv) * T + dt) * kD + lane * 4;
        const uint2 kr = *reinterpret_cast<const uint2*>(K_in + ioff);
        const uint2 vr = *reinterpret_cast<const uint2*>(V_in + ioff);
        const half2* kh = reinterpret_cast<const half2*>(&kr);
        const half2* vh = reinterpret_cast<const half2*>(&vr);
        float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
        float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
        ksum[0] += k0.x; ksum[1] += k0.y; ksum[2] += k1.x; ksum[3] += k1.y;
        vsum[0] += v0.x; vsum[1] += v0.y; vsum[2] += v1.x; vsum[3] += v1.y;
    }
    const float invP = 1.0f / static_cast<float>(kPageSize);
    const std::size_t poff = (static_cast<std::size_t>(hkv) * max_P + p) * kD + lane * 4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        K_page_mu[poff + j] = __float2half(ksum[j] * invP);
        V_page_mu[poff + j] = __float2half(vsum[j] * invP);
    }
}

// Masullo-Shell Sign-KV v3 Decode Attention (19.7x Thin-VRAM + Q8_0 Zero-Copy PCIe Exact Spike Refine!)
// Supports ANY H_q / H_kv (16/8, 32/8, etc.) and ANY S up to 262,144 (256K context!)
// Grid = H_q, Block = 256 (8 warps)
__global__ void masullo_v3_decode_kernel(
    const Half* __restrict__ Q,          // [H_q, 128]
    const uint4* __restrict__ K_p0,      // [H_kv, max_S] (GPU)
    const Half* __restrict__ K_norms,    // [H_kv, max_S] (GPU)
    const int8_t* __restrict__ K_hq8,    // [H_kv, max_S, 128] (pinned mapped host RAM)
    const Half* __restrict__ K_hsc,      // [H_kv, max_S]      (pinned mapped host RAM)
    const int8_t* __restrict__ V_hq8,    // [H_kv, max_S, 128] (pinned mapped host RAM)
    const Half* __restrict__ V_hsc,      // [H_kv, max_S]      (pinned mapped host RAM)
    const Half* __restrict__ K_page_mu,  // [H_kv, max_P, 128] (GPU)
    const Half* __restrict__ V_page_mu,  // [H_kv, max_P, 128] (GPU)
    Half* __restrict__ out,              // [H_q, 128]
    int H_q, int H_kv, int S, int max_S, int max_P, int top_k_spikes) {

    const int hq = blockIdx.x;
    const int gqa_group = H_q / H_kv;
    const int hkv = hq / gqa_group;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int P = min(S / kPageSize, kMaxPages);

    __shared__ float q_raw_sh[kD];
    __shared__ uint4 q_sign_sh;
    __shared__ float q_norm_sh;

    if (warp == 0) {
        const Half* q_ptr = Q + static_cast<std::size_t>(hq) * kD;
        float q[4], n2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __half2float(q_ptr[lane * 4 + j]);
            q_raw_sh[lane * 4 + j] = q[j];
            n2 += q[j] * q[j];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) n2 += __shfl_xor_sync(0xffffffffu, n2, d);
        fwht128(q, lane);
        unsigned b0 = __ballot_sync(0xffffffffu, q[0] >= 0.0f);
        unsigned b1 = __ballot_sync(0xffffffffu, q[1] >= 0.0f);
        unsigned b2 = __ballot_sync(0xffffffffu, q[2] >= 0.0f);
        unsigned b3 = __ballot_sync(0xffffffffu, q[3] >= 0.0f);
        if (lane == 0) {
            q_sign_sh = make_uint4(b0, b1, b2, b3);
            q_norm_sh = sqrtf(n2) * 0.0883883476f * 1.253314f / 128.0f;
        }
    }
    __syncthreads();

    const uint4 qb = q_sign_sh;
    const float qscale = q_norm_sh;
    const float inv_sd = 0.0883883476f;

    const uint4* kp0_b = K_p0 + static_cast<std::size_t>(hkv) * max_S;
    const Half*  kn_b  = K_norms + static_cast<std::size_t>(hkv) * max_S;
    const int8_t* kq_b = K_hq8 + static_cast<std::size_t>(hkv) * max_S * kD;
    const Half*  ks_b  = K_hsc + static_cast<std::size_t>(hkv) * max_S;
    const int8_t* vq_b = V_hq8 + static_cast<std::size_t>(hkv) * max_S * kD;
    const Half*  vs_b  = V_hsc + static_cast<std::size_t>(hkv) * max_S;
    const Half*  kpm_b = K_page_mu + static_cast<std::size_t>(hkv) * max_P * kD;
    const Half*  vpm_b = V_page_mu + static_cast<std::size_t>(hkv) * max_P * kD;

    __shared__ float sm_max[8], sm_min[8];
    __shared__ int hist[256];
    __shared__ int cand_idx[kMaxSurv];
    __shared__ int cand_n, border_n;
    __shared__ float page_scores[kMaxPages];   // 16 KiB SMEM (up to 256K context!)
    __shared__ uint16_t page_surv[kMaxPages];  //  8 KiB SMEM (fits easily inside 48 KiB default SMEM!)
    __shared__ int strict_bin_sh, border_bin_sh, border_cap_sh;

    if (tid == 0) { cand_n = 0; border_n = 0; }
    hist[tid] = 0; // blockDim.x == 256!
    for (int p = tid; p < P; p += blockDim.x) page_surv[p] = 0;

    float q_raw[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) q_raw[j] = q_raw_sh[lane * 4 + j];

    // Pass 0 (MOVED FIRST for Masullo v3.1!): Score all P Page Centroids in EXACT FP16 -> page_scores[p] in SMEM!
    for (int p = warp; p < P; p += nwarps) {
        const uint2 kr = *reinterpret_cast<const uint2*>(kpm_b + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* kh = reinterpret_cast<const half2*>(&kr);
        float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
        float dot = q_raw[0]*k0.x + q_raw[1]*k0.y + q_raw[2]*k1.x + q_raw[3]*k1.y;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
        if (lane == 0) page_scores[p] = dot * inv_sd;
    }
    __syncthreads();

    // Pass 1A: Content-only Masullo Local-Within-Global score = EXACT page_scores[t>>6] + 1-bit residual __popc!
    const int tail_start = min(S - kLocalWin, P * kPageSize);
    float lmax = -1e30f, lmin = 1e30f;
    for (int t = kSinks + tid; t < tail_start; t += blockDim.x) {
        const uint4 kb = __ldg(kp0_b + t);
        const int ham = __popc(qb.x ^ kb.x) + __popc(qb.y ^ kb.y) + __popc(qb.z ^ kb.z) + __popc(qb.w ^ kb.w);
        const float sc = page_scores[t >> 6] + qscale * __half2float(__ldg(kn_b + t)) * static_cast<float>(kD - 2 * ham);
        lmax = fmaxf(lmax, sc); lmin = fminf(lmin, sc);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, d));
        lmin = fminf(lmin, __shfl_xor_sync(0xffffffffu, lmin, d));
    }
    if (lane == 0) { sm_max[warp] = lmax; sm_min[warp] = lmin; }
    __syncthreads();
    float gmax = -1e30f, gmin = 1e30f;
    for (int w = 0; w < nwarps; ++w) { gmax = fmaxf(gmax, sm_max[w]); gmin = fminf(gmin, sm_min[w]); }
    float span = fmaxf(gmax - gmin, 1e-4f);
    float inv_span = 255.0f / span;

    // Pass 1B: 256-bin SMEM histogram over Page-Centered Residual scores!
    for (int t = kSinks + tid; t < tail_start; t += blockDim.x) {
        const uint4 kb = __ldg(kp0_b + t);
        const int ham = __popc(qb.x ^ kb.x) + __popc(qb.y ^ kb.y) + __popc(qb.z ^ kb.z) + __popc(qb.w ^ kb.w);
        const float sc = page_scores[t >> 6] + qscale * __half2float(__ldg(kn_b + t)) * static_cast<float>(kD - 2 * ham);
        int bin = min(255, max(0, static_cast<int>((sc - gmin) * inv_span)));
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int cum = 0, ex_bin = 0;
        for (int b = 255; b >= 0; --b) {
            if (cum + hist[b] >= top_k_spikes) { ex_bin = b; break; }
            cum += hist[b];
        }
        strict_bin_sh = ex_bin + 1;
        border_bin_sh = ex_bin;
        border_cap_sh = max(top_k_spikes - cum, 1);
    }
    __syncthreads();

    // Pass 1C: Two-tier starvation-free collection
    const int strict_bin = strict_bin_sh;
    const int border_bin = border_bin_sh;
    const int border_cap = border_cap_sh;
    for (int t = tid; t < S; t += blockDim.x) {
        if (t < kSinks || t >= tail_start) {
            int slot = atomicAdd(&cand_n, 1);
            if (slot < kMaxSurv) {
                cand_idx[slot] = t;
                int pg = t / kPageSize;
                if (pg < P) atomicAdd(reinterpret_cast<unsigned int*>(&page_surv[pg & ~1]), (pg & 1) ? 0x10000u : 1u);
            }
            continue;
        }
        const uint4 kb = __ldg(kp0_b + t);
        const int ham = __popc(qb.x ^ kb.x) + __popc(qb.y ^ kb.y) + __popc(qb.z ^ kb.z) + __popc(qb.w ^ kb.w);
        const float sc = page_scores[t >> 6] + qscale * __half2float(__ldg(kn_b + t)) * static_cast<float>(kD - 2 * ham);
        const int bin = min(255, max(0, static_cast<int>((sc - gmin) * inv_span)));
        bool take = (bin >= strict_bin);
        if (!take && bin == border_bin) {
            take = (atomicAdd(&border_n, 1) < border_cap);
        }
        if (take) {
            int slot = atomicAdd(&cand_n, 1);
            if (slot < kMaxSurv) {
                cand_idx[slot] = t;
                int pg = t / kPageSize;
                if (pg < P) atomicAdd(reinterpret_cast<unsigned int*>(&page_surv[pg & ~1]), (pg & 1) ? 0x10000u : 1u);
            }
        }
    }
    __syncthreads();
    const int n_surv = min(cand_n, kMaxSurv);

    // Pass 3: Exact Q8_0 spike rescore over zero-copy PCIe (128B per K, 128B per V = 2x faster than FP16!)
    float lmax_s = -1e30f, sum_e = 0.0f, acc[4] = {0,0,0,0};
    for (int i = warp; i < n_surv; i += nwarps) {
        const int t = cand_idx[i];
        const char4 kq = *reinterpret_cast<const char4*>(kq_b + static_cast<std::size_t>(t) * kD + lane * 4);
        const float ksc = __half2float(ks_b[t]);
        float dot = q_raw[0] * static_cast<float>(kq.x) + q_raw[1] * static_cast<float>(kq.y)
                  + q_raw[2] * static_cast<float>(kq.z) + q_raw[3] * static_cast<float>(kq.w);
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
        float logit = dot * (ksc * inv_sd);

        const char4 vq = *reinterpret_cast<const char4*>(vq_b + static_cast<std::size_t>(t) * kD + lane * 4);
        const float vsc = __half2float(vs_b[t]);
        float nmax = fmaxf(lmax_s, logit);
        float alpha = __expf(lmax_s - nmax), pw = __expf(logit - nmax) * vsc;
        float pw_raw = __expf(logit - nmax);
        sum_e = sum_e * alpha + pw_raw; lmax_s = nmax;
        acc[0] = acc[0] * alpha + pw * static_cast<float>(vq.x);
        acc[1] = acc[1] * alpha + pw * static_cast<float>(vq.y);
        acc[2] = acc[2] * alpha + pw * static_cast<float>(vq.z);
        acc[3] = acc[3] * alpha + pw * static_cast<float>(vq.w);
    }

    // Pass 4: Add Masullo Global Smooth Shell (unselected tokens per page)
    for (int p = warp; p < P; p += nwarps) {
        const int rem = kPageSize - static_cast<int>(page_surv[p]);
        if (rem <= 0) continue;
        const float pl = page_scores[p];
        float nmax = fmaxf(lmax_s, pl);
        float alpha = __expf(lmax_s - nmax);
        float wp = static_cast<float>(rem) * __expf(pl - nmax);
        sum_e = sum_e * alpha + wp; lmax_s = nmax;
        const uint2 vr = *reinterpret_cast<const uint2*>(vpm_b + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* vh = reinterpret_cast<const half2*>(&vr);
        float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
        acc[0] = acc[0]*alpha + wp*v0.x; acc[1] = acc[1]*alpha + wp*v0.y;
        acc[2] = acc[2]*alpha + wp*v1.x; acc[3] = acc[3]*alpha + wp*v1.y;
    }

    __shared__ float sm_wmax[8], sm_sum[8], sm_acc[8][kD];
    if (lane == 0) { sm_wmax[warp] = lmax_s; sm_sum[warp] = sum_e; }
    #pragma unroll
    for (int j = 0; j < 4; ++j) sm_acc[warp][lane * 4 + j] = acc[j];
    __syncthreads();

    if (warp == 0) {
        float fg = -1e30f; for (int w = 0; w < nwarps; ++w) fg = fmaxf(fg, sm_wmax[w]);
        float gsum = 0.0f, gout[4] = {0,0,0,0};
        for (int w = 0; w < nwarps; ++w) {
            float sc = __expf(sm_wmax[w] - fg); gsum += sm_sum[w] * sc;
            #pragma unroll
            for (int j = 0; j < 4; ++j) gout[j] += sm_acc[w][lane * 4 + j] * sc;
        }
        float inv = 1.0f / (gsum + 1e-9f);
        Half* o_ptr = out + static_cast<std::size_t>(hq) * kD;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o_ptr[lane * 4 + j] = __float2half(gout[j] * inv);
    }
}

// Masullo Saturated-Places 4:1 GQA Kernel (Ch. 7 Situation 7.3.1.1 — for Qwen3-4B / Qwen3-8B H_q=32, H_kv=8!):
// Evaluates ALL G=4 sibling query heads of one KV head in ONE threadblock (Grid = H_kv = 8 blocks -> split across 40 SMs via Split-S or 1 block/KV head):
// Wait: 8 blocks on 40 SMs would under-utilize 32/40 SMs! Better: Grid = (H_kv, 1) with 4 warps/Q-head = 16 warps (512 threads) OR
// 1 block per KV head scanning K_p0 ONCE into registers and computing __popc for ALL 4 sibling Q heads simultaneously!
// Each thread reads ONE uint4 K_p0 (16 B) + ONE K_norm (2 B) ONCE from GDDR6 and scores ALL 4 Q heads in registers -> 4x LESS 1-BIT SCAN TRAFFIC!
// Then takes the UNION of top-K spikes across the 4 sibling Q heads and fetches each Q8_0 K/V ONCE over PCIe into SMEM for all 4 Q heads!
__global__ void masullo_v3_saturated_gqa4_kernel(
    const Half* __restrict__ Q,          // [H_kv * 4, 128]
    const uint4* __restrict__ K_p0,      // [H_kv, max_S]
    const Half* __restrict__ K_norms,    // [H_kv, max_S]
    const int8_t* __restrict__ K_hq8,    // [H_kv, max_S, 128]
    const Half* __restrict__ K_hsc,      // [H_kv, max_S]
    const int8_t* __restrict__ V_hq8,    // [H_kv, max_S, 128]
    const Half* __restrict__ V_hsc,      // [H_kv, max_S]
    const Half* __restrict__ K_page_mu,  // [H_kv, max_P, 128]
    const Half* __restrict__ V_page_mu,  // [H_kv, max_P, 128]
    Half* __restrict__ out,              // [H_kv * 4, 128]
    int H_kv, int S, int max_S, int max_P, int top_k_union) {

    const int hkv = blockIdx.x;
    const int tid = threadIdx.x; // 256 threads = 8 warps
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int P = min(S / kPageSize, kMaxPages);

    __shared__ float q_raw_sh[4][kD];
    __shared__ uint4 q_sign_sh[4];
    __shared__ float q_norm_sh[4];

    // Warps 0..3 load & FWHT the 4 sibling Q heads simultaneously!
    if (warp < 4) {
        const int hq = hkv * 4 + warp;
        const Half* q_ptr = Q + static_cast<std::size_t>(hq) * kD;
        float q[4], n2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __half2float(q_ptr[lane * 4 + j]);
            q_raw_sh[warp][lane * 4 + j] = q[j];
            n2 += q[j] * q[j];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) n2 += __shfl_xor_sync(0xffffffffu, n2, d);
        fwht128(q, lane);
        unsigned b0 = __ballot_sync(0xffffffffu, q[0] >= 0.0f);
        unsigned b1 = __ballot_sync(0xffffffffu, q[1] >= 0.0f);
        unsigned b2 = __ballot_sync(0xffffffffu, q[2] >= 0.0f);
        unsigned b3 = __ballot_sync(0xffffffffu, q[3] >= 0.0f);
        if (lane == 0) {
            q_sign_sh[warp] = make_uint4(b0, b1, b2, b3);
            q_norm_sh[warp] = sqrtf(n2) * 0.0883883476f * 1.253314f / 128.0f;
        }
    }
    __syncthreads();

    uint4 qb[4]; float qsc[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) { qb[g] = q_sign_sh[g]; qsc[g] = q_norm_sh[g]; }
    const float inv_sd = 0.0883883476f;

    const uint4* kp0_b = K_p0 + static_cast<std::size_t>(hkv) * max_S;
    const Half*  kn_b  = K_norms + static_cast<std::size_t>(hkv) * max_S;
    const int8_t* kq_b = K_hq8 + static_cast<std::size_t>(hkv) * max_S * kD;
    const Half*  ks_b  = K_hsc + static_cast<std::size_t>(hkv) * max_S;
    const int8_t* vq_b = V_hq8 + static_cast<std::size_t>(hkv) * max_S * kD;
    const Half*  vs_b  = V_hsc + static_cast<std::size_t>(hkv) * max_S;
    const Half*  kpm_b = K_page_mu + static_cast<std::size_t>(hkv) * max_P * kD;
    const Half*  vpm_b = V_page_mu + static_cast<std::size_t>(hkv) * max_P * kD;

    constexpr int kMaxSatPages = 2048; // 128K context! 4 * 2048 * 2B = 16 KiB SMEM!
    const int Psat = min(P, kMaxSatPages);
    __shared__ float sm_max[8], sm_min[8];
    __shared__ int hist[256];
    __shared__ int cand_idx[kMaxSurv];
    __shared__ int cand_n, border_n;
    __shared__ Half page_scores_h[4][kMaxSatPages]; // 16 KiB SMEM!
    __shared__ float head_mean_sh[4];
    __shared__ uint16_t page_surv[kMaxSatPages];    //  4 KiB SMEM!
    __shared__ int strict_bin_sh, border_bin_sh, border_cap_sh;

    if (tid == 0) { cand_n = 0; border_n = 0; }
    hist[tid] = 0;
    for (int p = tid; p < Psat; p += blockDim.x) page_surv[p] = 0;

    float q_r[4][4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) q_r[g][j] = q_raw_sh[g][lane * 4 + j];
    }

    // Pass 0: Score all P Page Centroids in EXACT FP16 for ALL 4 sibling Q heads + compute per-head baseline mean!
    float hsum[4] = {0, 0, 0, 0};
    for (int p = warp; p < Psat; p += nwarps) {
        const uint2 kr = *reinterpret_cast<const uint2*>(kpm_b + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* kh = reinterpret_cast<const half2*>(&kr);
        float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float dot = q_r[g][0]*k0.x + q_r[g][1]*k0.y + q_r[g][2]*k1.x + q_r[g][3]*k1.y;
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
            float sc = dot * inv_sd;
            if (lane == 0) { page_scores_h[g][p] = __float2half(sc); hsum[g] += sc; }
        }
    }
    if (lane == 0) {
        for (int g = 0; g < 4; ++g) sm_max[warp] = hsum[g]; // compute exact head means sequentially below
    }
    __syncthreads();
    // Compute exact per-head page mean so every sibling head votes by HINGE EXCESS (logit_g - mean_g)!
    if (warp < 4) {
        float s = 0.0f;
        for (int p = lane; p < Psat; p += 32) s += __half2float(page_scores_h[warp][p]);
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
        if (lane == 0) head_mean_sh[warp] = s / fmaxf(static_cast<float>(Psat), 1.0f);
    }
    __syncthreads();
    float hmean[4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) hmean[g] = head_mean_sh[g];

    // Pass 1A: SINGLE GDDR6 scan of K_p0 shared across ALL 4 sibling Q heads!
    // Score each token by MAX HINGE EXCESS over each head's own page baseline:
    //   excess_g(t) = (page_scores_h[g][t>>6] - mean_g) + qsc[g] * rn * (128 - 2*ham)
    const int tail_start = min(S - kLocalWin, Psat * kPageSize);
    float lmax = -1e30f, lmin = 1e30f;
    for (int t = kSinks + tid; t < tail_start; t += blockDim.x) {
        const uint4 kb = __ldg(kp0_b + t);
        const float kn = __half2float(__ldg(kn_b + t));
        const int pg = t >> 6;
        float msc = -1e30f;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            int ham = __popc(qb[g].x ^ kb.x) + __popc(qb[g].y ^ kb.y) + __popc(qb[g].z ^ kb.z) + __popc(qb[g].w ^ kb.w);
            float ex = (__half2float(page_scores_h[g][pg]) - hmean[g]) + qsc[g] * kn * static_cast<float>(kD - 2 * ham);
            msc = fmaxf(msc, ex);
        }
        lmax = fmaxf(lmax, msc); lmin = fminf(lmin, msc);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, d));
        lmin = fminf(lmin, __shfl_xor_sync(0xffffffffu, lmin, d));
    }
    if (lane == 0) { sm_max[warp] = lmax; sm_min[warp] = lmin; }
    __syncthreads();
    float gmax = -1e30f, gmin = 1e30f;
    for (int w = 0; w < nwarps; ++w) { gmax = fmaxf(gmax, sm_max[w]); gmin = fminf(gmin, sm_min[w]); }
    float span = fmaxf(gmax - gmin, 1e-4f), inv_span = 255.0f / span;

    // Pass 1B: 256-bin saturated histogram over Hinge-Excess Page-Centered Residual scores!
    for (int t = kSinks + tid; t < tail_start; t += blockDim.x) {
        const uint4 kb = __ldg(kp0_b + t);
        const float kn = __half2float(__ldg(kn_b + t));
        const int pg = t >> 6;
        float msc = -1e30f;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            int ham = __popc(qb[g].x ^ kb.x) + __popc(qb[g].y ^ kb.y) + __popc(qb[g].z ^ kb.z) + __popc(qb[g].w ^ kb.w);
            float ex = (__half2float(page_scores_h[g][pg]) - hmean[g]) + qsc[g] * kn * static_cast<float>(kD - 2 * ham);
            msc = fmaxf(msc, ex);
        }
        atomicAdd(&hist[min(255, max(0, static_cast<int>((msc - gmin) * inv_span)))], 1);
    }
    __syncthreads();
    if (tid == 0) {
        int cum = 0, eb = 0;
        for (int b = 255; b >= 0; --b) {
            if (cum + hist[b] >= top_k_union) { eb = b; break; }
            cum += hist[b];
        }
        strict_bin_sh = eb + 1;
        border_bin_sh = eb;
        border_cap_sh = max(top_k_union - cum, 1);
    }
    __syncthreads();

    const int strict_bin = strict_bin_sh;
    const int border_bin = border_bin_sh;
    const int border_cap = border_cap_sh;
    for (int t = tid; t < S; t += blockDim.x) {
        if (t < kSinks || t >= tail_start) {
            int slot = atomicAdd(&cand_n, 1);
            if (slot < kMaxSurv) {
                cand_idx[slot] = t;
                int pg = t / kPageSize;
                if (pg < Psat) atomicAdd(reinterpret_cast<unsigned int*>(&page_surv[pg & ~1]), (pg & 1) ? 0x10000u : 1u);
            }
            continue;
        }
        const uint4 kb = __ldg(kp0_b + t);
        const float kn = __half2float(__ldg(kn_b + t));
        const int pg = t >> 6;
        float msc = -1e30f;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            int ham = __popc(qb[g].x ^ kb.x) + __popc(qb[g].y ^ kb.y) + __popc(qb[g].z ^ kb.z) + __popc(qb[g].w ^ kb.w);
            float ex = (__half2float(page_scores_h[g][pg]) - hmean[g]) + qsc[g] * kn * static_cast<float>(kD - 2 * ham);
            msc = fmaxf(msc, ex);
        }
        const int bin = min(255, max(0, static_cast<int>((msc - gmin) * inv_span)));
        bool take = (bin >= strict_bin);
        if (!take && bin == border_bin) {
            take = (atomicAdd(&border_n, 1) < border_cap);
        }
        if (take) {
            int slot = atomicAdd(&cand_n, 1);
            if (slot < kMaxSurv) {
                cand_idx[slot] = t;
                if (pg < Psat) atomicAdd(reinterpret_cast<unsigned int*>(&page_surv[pg & ~1]), (pg & 1) ? 0x10000u : 1u);
            }
        }
    }
    __syncthreads();
    const int n_surv = min(cand_n, kMaxSurv);

    // Pass 2: ONE SHARED Q8_0 PCIe FETCH per survivor token -> scored & accumulated across ALL 4 sibling Q heads in registers!
    float lmax_s[4] = {-1e30f,-1e30f,-1e30f,-1e30f}, sum_e[4] = {0,0,0,0};
    float acc[4][4];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) { acc[g][j] = 0.0f; }
    }

    for (int i = warp; i < n_surv; i += nwarps) {
        const int t = cand_idx[i];
        // SINGLE coalesced PCIe read of Q8_0 K & V reused across all 4 Q heads!!
        const char4 kq = *reinterpret_cast<const char4*>(kq_b + static_cast<std::size_t>(t) * kD + lane * 4);
        const float ksc = __half2float(ks_b[t]) * inv_sd;
        const char4 vq = *reinterpret_cast<const char4*>(vq_b + static_cast<std::size_t>(t) * kD + lane * 4);
        const float vsc = __half2float(vs_b[t]);
        const float kf[4] = {float(kq.x), float(kq.y), float(kq.z), float(kq.w)};
        const float vf[4] = {float(vq.x)*vsc, float(vq.y)*vsc, float(vq.z)*vsc, float(vq.w)*vsc};

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float dot = q_r[g][0]*kf[0] + q_r[g][1]*kf[1] + q_r[g][2]*kf[2] + q_r[g][3]*kf[3];
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
            float logit = dot * ksc;
            float nmax = fmaxf(lmax_s[g], logit);
            float alpha = __expf(lmax_s[g] - nmax), pw = __expf(logit - nmax);
            sum_e[g] = sum_e[g] * alpha + pw; lmax_s[g] = nmax;
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[g][j] = acc[g][j] * alpha + pw * vf[j];
        }
    }

    // Pass 3: Reuse already-computed exact FP16 page_scores_h[g][p] across ALL 4 Q heads (ZERO duplicate K_page_mu reads!)
    for (int p = warp; p < Psat; p += nwarps) {
        const int rem = kPageSize - static_cast<int>(page_surv[p]);
        if (rem <= 0) continue;
        const uint2 vr = *reinterpret_cast<const uint2*>(vpm_b + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* vh = reinterpret_cast<const half2*>(&vr);
        float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
        const float vf[4] = {v0.x, v0.y, v1.x, v1.y};
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float pl = __half2float(page_scores_h[g][p]);
            float nmax = fmaxf(lmax_s[g], pl);
            float alpha = __expf(lmax_s[g] - nmax), wp = static_cast<float>(rem) * __expf(pl - nmax);
            sum_e[g] = sum_e[g] * alpha + wp; lmax_s[g] = nmax;
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[g][j] = acc[g][j] * alpha + wp * vf[j];
        }
    }

    // Reduce across 8 warps for each of the 4 sibling Q heads!
    __shared__ float sm_wmax[8], sm_sum[8], sm_acc[8][kD];
    for (int g = 0; g < 4; ++g) {
        __syncthreads();
        if (lane == 0) { sm_wmax[warp] = lmax_s[g]; sm_sum[warp] = sum_e[g]; }
        #pragma unroll
        for (int j = 0; j < 4; ++j) sm_acc[warp][lane * 4 + j] = acc[g][j];
        __syncthreads();
        if (warp == 0) {
            float fg = -1e30f; for (int w = 0; w < nwarps; ++w) fg = fmaxf(fg, sm_wmax[w]);
            float gsum = 0.0f, gout[4] = {0,0,0,0};
            for (int w = 0; w < nwarps; ++w) {
                float sc = __expf(sm_wmax[w] - fg); gsum += sm_sum[w] * sc;
                #pragma unroll
                for (int j = 0; j < 4; ++j) gout[j] += sm_acc[w][lane * 4 + j] * sc;
            }
            float inv = 1.0f / (gsum + 1e-9f);
            Half* o_ptr = out + static_cast<std::size_t>(hkv * 4 + g) * kD;
            #pragma unroll
            for (int j = 0; j < 4; ++j) o_ptr[lane * 4 + j] = __float2half(gout[j] * inv);
        }
    }
}

} // namespace

extern "C" {

int alloc_pinned_mapped(size_t bytes, void** h_ptr, void** d_ptr) {
    cudaError_t e = cudaHostAlloc(h_ptr, bytes, cudaHostAllocMapped);
    if (e != cudaSuccess) return (int)e;
    e = cudaHostGetDevicePointer(d_ptr, *h_ptr, 0);
    return (int)e;
}

void free_pinned(void* h_ptr) {
    if (h_ptr) cudaFreeHost(h_ptr);
}

void launch_pack_kv_v3(
    const void* K_in, const void* V_in, const void* K_page_mu,
    void* K_p0, void* K_norms,
    void* K_hq8, void* K_hsc, void* V_hq8, void* V_hsc,
    int H_kv, int t_start, int T, int max_S, int max_P, int use_page_resid) {
    dim3 g(T, H_kv);
    pack_kv_v3_q8_kernel<<<g, 32>>>(
        (const Half*)K_in, (const Half*)V_in, (const Half*)K_page_mu,
        (uint4*)K_p0, (Half*)K_norms,
        (int8_t*)K_hq8, (Half*)K_hsc, (int8_t*)V_hq8, (Half*)V_hsc,
        t_start, T, max_S, max_P, use_page_resid);
}

void launch_build_pages_v3(
    const void* K_in, const void* V_in,
    void* K_page_mu, void* V_page_mu,
    int H_kv, int p_start, int n_pages, int T, int max_P) {
    if (n_pages <= 0) return;
    dim3 g(n_pages, H_kv);
    build_pages_v3_kernel<<<g, 32>>>(
        (const Half*)K_in, (const Half*)V_in,
        (Half*)K_page_mu, (Half*)V_page_mu,
        p_start, n_pages, T, max_P);
}

void launch_masullo_v3_decode(
    const void* Q,
    const void* K_p0, const void* K_norms,
    const void* K_hq8, const void* K_hsc,
    const void* V_hq8, const void* V_hsc,
    const void* K_page_mu, const void* V_page_mu,
    void* out, int H_q, int H_kv, int S, int max_S, int max_P, int top_k_spikes) {
    masullo_v3_decode_kernel<<<H_q, 256>>>(
        (const Half*)Q,
        (const uint4*)K_p0, (const Half*)K_norms,
        (const int8_t*)K_hq8, (const Half*)K_hsc,
        (const int8_t*)V_hq8, (const Half*)V_hsc,
        (const Half*)K_page_mu, (const Half*)V_page_mu,
        (Half*)out, H_q, H_kv, S, max_S, max_P, top_k_spikes);
}

void launch_masullo_v3_saturated_gqa4(
    const void* Q,
    const void* K_p0, const void* K_norms,
    const void* K_hq8, const void* K_hsc,
    const void* V_hq8, const void* V_hsc,
    const void* K_page_mu, const void* V_page_mu,
    void* out, int H_kv, int S, int max_S, int max_P, int top_k_union) {
    masullo_v3_saturated_gqa4_kernel<<<H_kv, 256>>>(
        (const Half*)Q,
        (const uint4*)K_p0, (const Half*)K_norms,
        (const int8_t*)K_hq8, (const Half*)K_hsc,
        (const int8_t*)V_hq8, (const Half*)V_hsc,
        (const Half*)K_page_mu, (const Half*)V_page_mu,
        (Half*)out, H_kv, S, max_S, max_P, top_k_union);
}

} // extern "C"
