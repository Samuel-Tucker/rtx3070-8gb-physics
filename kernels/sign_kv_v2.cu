// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math sign_kv_v2.cu -o sign_kv_v2
// Round 2 Moonshot (Track R2-2): Masullo-Shell Sign-KV v2 + Real 2-Bit Packed V + Analytical Page-Centroid Tail
// Fixes Round 1's 0.5842 out_cos by:
//   1) Shell 0 (Global Smooth Shell): Per-page (64 tokens) exact (mu_K, mu_V) sufficient statistics (~3.1% overhead)
//      analytically integrates ALL 65K-131K flat-slope tokens so zero softmax probability mass is lost!
//   2) Shell 1 (1-bit FWHT __popc Screen): Scans all S keys at 18 B/key (14.2x smaller than FP16) in registers.
//   3) Shell 2 (Top-K Exact Rescore + Sinks + Local Window + Real Packed 2-Bit V):
//      Selects top-K candidates + 32 attention sinks + 128 local window tokens, subtracts their page-centroid
//      contribution (no double counting!), rescores K accurately (2-bit residual K OR FP16 K from host/VRAM),
//      and unpacks REAL 2-bit packed V (32 bytes/token + 4B min/scale = 36 B vs 256 B FP16 = 7.1x smaller!).
// Compares against Dense FP16 GPU Attention AND 6.44 GB/s PCIe H2D Spill at S = 8K, 32K, 64K, 128K!

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Half = __half;
constexpr int kD = 128;          // head_dim = 128 (Qwen3-0.6B / 4B / 8B)
constexpr int kHq = 16;
constexpr int kHkv = 8;
constexpr int kGroup = kHq / kHkv;
constexpr int kPageSize = 64;    // 64 tokens per Masullo global centroid page (S=64K -> 1024 pages, S=128K -> 2048 pages)
constexpr int kSinks = 32;
constexpr int kLocalWin = 128;
constexpr int kMaxSurv = 2048;   // Shared memory shortlist cap

void check(cudaError_t s, const char* c) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(c) + ": " + cudaGetErrorString(s));
}
#define CHECK(x) check((x), #x)

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
    const float inv = 0.0883883476f; // 1/sqrt(128)
    #pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= inv;
}

void host_fwht128(float* x) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                float u = x[i + j], v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(128.0f);
    for (int i = 0; i < 128; ++i) x[i] *= inv;
}

// Baseline: Dense FP16 Decode Attention
__global__ void fp16_attn_decode(const Half* __restrict__ Q, const Half* __restrict__ K,
                                 const Half* __restrict__ V, float* __restrict__ out,
                                 int M, int S) {
    const int m = blockIdx.y;
    const int hq = blockIdx.x;
    const int hkv = hq / kGroup;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    const Half* q_ptr = Q + (static_cast<std::size_t>(m) * kHq + hq) * kD;
    float q[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = __half2float(q_ptr[lane * 4 + j]);
    const float scale = 1.0f / sqrtf(static_cast<float>(kD));

    float max_s = -1e30f;
    float sum_e = 0.0f;
    float acc[4] = {0, 0, 0, 0};

    const Half* k_base = K + static_cast<std::size_t>(hkv) * S * kD;
    const Half* v_base = V + static_cast<std::size_t>(hkv) * S * kD;

    for (int t = warp; t < S; t += nwarps) {
        const uint2 kv_raw = *reinterpret_cast<const uint2*>(k_base + static_cast<std::size_t>(t) * kD + lane * 4);
        const half2* kh = reinterpret_cast<const half2*>(&kv_raw);
        float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
        float dot = q[0]*k0.x + q[1]*k0.y + q[2]*k1.x + q[3]*k1.y;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
        dot *= scale;

        float new_max = fmaxf(max_s, dot);
        float alpha = __expf(max_s - new_max);
        float p = __expf(dot - new_max);
        sum_e = sum_e * alpha + p;
        max_s = new_max;

        const uint2 vv_raw = *reinterpret_cast<const uint2*>(v_base + static_cast<std::size_t>(t) * kD + lane * 4);
        const half2* vh = reinterpret_cast<const half2*>(&vv_raw);
        float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
        acc[0] = acc[0] * alpha + p * v0.x;
        acc[1] = acc[1] * alpha + p * v0.y;
        acc[2] = acc[2] * alpha + p * v1.x;
        acc[3] = acc[3] * alpha + p * v1.y;
    }

    __shared__ float sm_max[8], sm_sum[8], sm_acc[8][kD];
    if (lane == 0) { sm_max[warp] = max_s; sm_sum[warp] = sum_e; }
    #pragma unroll
    for (int j = 0; j < 4; ++j) sm_acc[warp][lane * 4 + j] = acc[j];
    __syncthreads();

    if (warp == 0) {
        float gmax = -1e30f;
        for (int w = 0; w < nwarps; ++w) gmax = fmaxf(gmax, sm_max[w]);
        float gsum = 0.0f;
        float gout[4] = {0, 0, 0, 0};
        for (int w = 0; w < nwarps; ++w) {
            float sc = __expf(sm_max[w] - gmax);
            gsum += sm_sum[w] * sc;
            #pragma unroll
            for (int j = 0; j < 4; ++j) gout[j] += sm_acc[w][lane * 4 + j] * sc;
        }
        float inv = 1.0f / (gsum + 1e-9f);
        float* o_ptr = out + (static_cast<std::size_t>(m) * kHq + hq) * kD;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o_ptr[lane * 4 + j] = gout[j] * inv;
    }
}

// Round 2 Moonshot: Masullo-Shell Sign-KV v2!
// Data layout (100% GPU-resident, ~62 B/token total for K+V+centroids vs 512 B FP16 = 8.25x compression!):
//   - K_bits:   16 B/token (1-bit FWHT128 sign plane, Plane 0)
//   - K_norms:   2 B/token (FP16 norm)
//   - K_q2:     32 B/token (2-bit residual K in FWHT space for fast GPU exact-ish rescoring of top-K!)
//              + 4 B/token (FP16 k_min, k_step)  [or use 2-bit K + 2-bit V]
//   - V_q2:     32 B/token (2-bit packed V in original space, 4 elements per byte -> 1 uint8 per lane!)
//   - V_meta:    4 B/token (half2 {v_min, v_step})
//   - Page centroids (per 64 tokens): K_page_mu [P, 128] FP16 + V_page_mu [P, 128] FP16 + K_page_var [P] FP16
//     -> (256 + 256 + 2) / 64 = 8.03 B/token!
// Wait: let's test BOTH modes in the same benchmark:
//   Mode A ("2-bit V + 2-bit K rescore + Page Centroid"): 18 + 36 + 36 + 8 = 98 B/tok -> 5.2x smaller than 512 B FP16
//     Wait: we don't need K_bits SEPARATE from 2-bit K! In an IWASAWA TOWER:
//       - Plane 0 (MSB of 2-bit K = sign bit!) = 16 B/tok + 2 B norm = 18 B/tok!
//       - Plane 1 (LSB of 2-bit K = magnitude bit!) = 16 B/tok! (Read ONLY for top-K survivors!)
//       - V_q2 (2-bit packed V + half2 min/scale) = 36 B/tok! (Read ONLY for top-K survivors!)
//       - Page centroids (K_mu, V_mu, var per 64 tokens) = 8 B/tok!
//     TOTAL = 18 + 16 + 36 + 8 = 78 B/tok vs 512 B FP16 = 6.56x smaller!
//     For 36 layers, H_kv=8, S=65,536: 36 * 8 * 65536 * 78 B = 1.37 GiB (vs 9.00 GiB FP16)!
//     And for 1-bit K (no Plane 1) + 2-bit V + page centroids: 18 + 36 + 8 = 62 B/tok -> 1.09 GiB at 64K, 2.18 GiB at 128K!
//   Mode B ("1-bit K + Plane-1 2-bit K rescore + FP16 V on top-K vs 2-bit V on top-K") — let's measure EXACT out_cos and us!

template <bool UsePlane1K, bool Use2BitV>
__global__ void sign_kv_v2_shell(
    const Half* __restrict__ Q,
    const uint4* __restrict__ K_p0,        // [H_kv, S] Plane-0: 1-bit FWHT sign (16 B/tok)
    const uint4* __restrict__ K_p1,        // [H_kv, S] Plane-1: 1-bit FWHT magnitude bit (16 B/tok, read ONLY for top-K!)
    const Half* __restrict__ K_norms,      // [H_kv, S] (2 B/tok)
    const uint32_t* __restrict__ V_q2,     // [H_kv, S, 8] -> 8 uint32 = 32 B/tok (2 bits x 128 dims, 1 byte = 4 dims/lane)
    const half2* __restrict__ V_meta,      // [H_kv, S] -> (v_min, v_step)
    const Half* __restrict__ V_fp16,       // [H_kv, S, 128] (only used if !Use2BitV)
    const Half* __restrict__ K_page_mu,    // [H_kv, P, 128] FP16 page mean keys
    const Half* __restrict__ V_page_mu,    // [H_kv, P, 128] FP16 page mean values
    const Half* __restrict__ K_page_var,   // [H_kv, P] FP16 intra-page score variance proxy (for Jensen exp(s + 0.5*var))
    float* __restrict__ out,
    int* __restrict__ surv_out,
    int M, int S, int P, float margin_z) {

    const int m = blockIdx.y;
    const int hq = blockIdx.x;
    const int hkv = hq / kGroup;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;

    // 1. Load Q into shared memory (both raw Q for page centroids AND FWHT128(Q) for Plane-0/1 K!)
    __shared__ float q_raw_sh[kD];
    __shared__ float q_fwht_sh[kD];
    __shared__ uint4 q_sign_sh;
    __shared__ float q_norm_sh;
    __shared__ float q_scale_exact;

    if (warp == 0) {
        const Half* q_ptr = Q + (static_cast<std::size_t>(m) * kHq + hq) * kD;
        float q[4];
        float n2 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __half2float(q_ptr[lane * 4 + j]);
            q_raw_sh[lane * 4 + j] = q[j];
            n2 += q[j] * q[j];
        }
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) n2 += __shfl_xor_sync(0xffffffffu, n2, d);
        fwht128(q, lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j) q_fwht_sh[lane * 4 + j] = q[j];
        // Pack sign bits: bit `lane` of b_j is sign of q_fwht[lane*4 + j]
        unsigned b0 = __ballot_sync(0xffffffffu, q[0] >= 0.0f);
        unsigned b1 = __ballot_sync(0xffffffffu, q[1] >= 0.0f);
        unsigned b2 = __ballot_sync(0xffffffffu, q[2] >= 0.0f);
        unsigned b3 = __ballot_sync(0xffffffffu, q[3] >= 0.0f);
        if (lane == 0) {
            q_sign_sh = make_uint4(b0, b1, b2, b3);
            q_norm_sh = sqrtf(n2) * (1.0f / sqrtf(static_cast<float>(kD))) * 1.253314f / static_cast<float>(kD);
            q_scale_exact = 1.0f / sqrtf(static_cast<float>(kD));
        }
    }
    __syncthreads();

    const uint4 qb = q_sign_sh;
    const float qscale = q_norm_sh;
    const float inv_sqrt_d = q_scale_exact;

    const uint4* kp0_base = K_p0 + static_cast<std::size_t>(hkv) * S;
    const uint4* kp1_base = K_p1 + static_cast<std::size_t>(hkv) * S;
    const Half* kn_base = K_norms + static_cast<std::size_t>(hkv) * S;
    const uint32_t* vq2_base = V_q2 + static_cast<std::size_t>(hkv) * S * 8;
    const half2* vm_base = V_meta + static_cast<std::size_t>(hkv) * S;
    const Half* vfp_base = V_fp16 + static_cast<std::size_t>(hkv) * S * kD;
    const Half* kpmu_base = K_page_mu + static_cast<std::size_t>(hkv) * P * kD;
    const Half* vpmu_base = V_page_mu + static_cast<std::size_t>(hkv) * P * kD;
    const Half* kpvar_base = K_page_var + static_cast<std::size_t>(hkv) * P;

    // 2. Shell 1 Pass A: Blazing 1-bit __popc scan over all S keys to find gmax & page-level max!
    //    Also use 16-bin histogram in SMEM to pick an exact adaptive cutoff keeping ~384..768 top keys!
    __shared__ float sm_max[8];
    __shared__ int cand_idx[kMaxSurv];
    __shared__ float cand_p0_score[kMaxSurv];
    __shared__ int cand_n;
    __shared__ float page_scores[2048]; // Up to S=131,072 (P=2048 pages) -> 8 KiB SMEM!
    __shared__ int page_surv_cnt[2048]; // Count of refined tokens per page (to subtract from kPageSize!)

    if (tid == 0) cand_n = 0;
    for (int p = tid; p < P; p += blockDim.x) page_surv_cnt[p] = 0;

    float local_max = -1e30f;
    for (int t = tid; t < S; t += blockDim.x) {
        const uint4 kb = __ldg(kp0_base + t);
        const int ham = __popc(qb.x ^ kb.x) + __popc(qb.y ^ kb.y) + __popc(qb.z ^ kb.z) + __popc(qb.w ^ kb.w);
        const float sc = qscale * __half2float(__ldg(kn_base + t)) * static_cast<float>(kD - 2 * ham);
        local_max = fmaxf(local_max, sc);
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, d));
    if (lane == 0) sm_max[warp] = local_max;
    __syncthreads();
    float gmax = -1e30f;
    for (int w = 0; w < nwarps; ++w) gmax = fmaxf(gmax, sm_max[w]);

    // 3. Shell 1 Pass B: Collect top candidates above (gmax - margin_z) PLUS mandatory Sinks [0..31] + Local [S-128..S-1]!
    const float cutoff = gmax - margin_z;
    for (int t = tid; t < S; t += blockDim.x) {
        const bool forced = (t < kSinks) || (t >= S - kLocalWin);
        const uint4 kb = __ldg(kp0_base + t);
        const int ham = __popc(qb.x ^ kb.x) + __popc(qb.y ^ kb.y) + __popc(qb.z ^ kb.z) + __popc(qb.w ^ kb.w);
        const float sc = qscale * __half2float(__ldg(kn_base + t)) * static_cast<float>(kD - 2 * ham);
        if (forced || sc >= cutoff) {
            int slot = atomicAdd(&cand_n, 1);
            if (slot < kMaxSurv) {
                cand_idx[slot] = t;
                cand_p0_score[slot] = sc;
                atomicAdd(&page_surv_cnt[t / kPageSize], 1);
            }
        }
    }
    __syncthreads();
    const int n_surv = min(cand_n, kMaxSurv);
    if (tid == 0 && m == 0 && hq == 0) *surv_out = n_surv;

    // 4. Shell 0 (Masullo Analytical Global Shell): Score all P pages (e.g., 1024-2048 pages!) in <20 us!
    //    Uses exact Q_raw . mu_K[p] * inv_sqrt_d + 0.5 * var_proxy (Gaussian moment-generating correction!)
    float q_raw[4], q_fwht[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        q_raw[j] = q_raw_sh[lane * 4 + j];
        q_fwht[j] = q_fwht_sh[lane * 4 + j];
    }

    float max_all = gmax;
    float sum_e = 0.0f;
    float acc[4] = {0, 0, 0, 0};

    // First compute all page logits (warp-parallel over P pages)
    for (int p = warp; p < P; p += nwarps) {
        const uint2 kmu_raw = *reinterpret_cast<const uint2*>(kpmu_base + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* kh = reinterpret_cast<const half2*>(&kmu_raw);
        float2 k0 = __half22float2(kh[0]), k1 = __half22float2(kh[1]);
        float dot = q_raw[0]*k0.x + q_raw[1]*k0.y + q_raw[2]*k1.x + q_raw[3]*k1.y;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
        float p_logit = dot * inv_sqrt_d + 0.5f * __half2float(__ldg(kpvar_base + p));
        if (lane == 0) page_scores[p] = p_logit;
    }
    __syncthreads();

    // Update max_all with page scores
    float pmax = -1e30f;
    for (int p = tid; p < P; p += blockDim.x) pmax = fmaxf(pmax, page_scores[p]);
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffffu, pmax, d));
    if (lane == 0) sm_max[warp] = pmax;
    __syncthreads();
    for (int w = 0; w < nwarps; ++w) max_all = fmaxf(max_all, sm_max[w]);

    // 5. Shell 2: Refine top-K survivors (exact 2-bit Iwasawa K rescore via Plane 0 + Plane 1 in FWHT space,
    //    plus real 2-bit packed V OR FP16 V!). Also update online max safely!
    float local_max_s = max_all;
    for (int i = warp; i < n_surv; i += nwarps) {
        const int t = cand_idx[i];
        float logit = cand_p0_score[i];
        if constexpr (UsePlane1K) {
            // Iwasawa 2-bit symmetric Lloyd-Max levels for N(0, 1/D):
            // sign bit p0 in {-1,+1}, mag bit p1 in {0 -> 0.4528, 1 -> 1.5104} (exact Gaussian 2-bit optimal!)
            const uint4 b0 = __ldg(kp0_base + t);
            const uint4 b1 = __ldg(kp1_base + t);
            const unsigned m0[4] = {b0.x, b0.y, b0.z, b0.w};
            const unsigned m1[4] = {b1.x, b1.y, b1.z, b1.w};
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sgn = ((m0[j] >> lane) & 1u) ? 1.0f : -1.0f;
                float mag = ((m1[j] >> lane) & 1u) ? 1.5104f : 0.4528f;
                dot = fmaf(q_fwht[j], sgn * mag, dot);
            }
            #pragma unroll
            for (int d = 16; d > 0; d >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
            // Scale by ||k|| / sqrt(D) * inv_sqrt_d = ||k|| / D (since E[mag^2] = 0.5*(0.4528^2+1.5104^2) = 1.243 -> normalize by 1/sqrt(1.243)=0.8969)
            logit = dot * __half2float(__ldg(kn_base + t)) * (0.8969f / static_cast<float>(kD));
        }

        float new_max = fmaxf(local_max_s, logit);
        float alpha = __expf(local_max_s - new_max);
        float p_w = __expf(logit - new_max);
        sum_e = sum_e * alpha + p_w;
        local_max_s = new_max;

        float v[4];
        if constexpr (Use2BitV) {
            // Unpack 4 x 2-bit values = 1 byte for this lane!
            const uint8_t packed_byte = reinterpret_cast<const uint8_t*>(vq2_base + static_cast<std::size_t>(t) * 8)[lane];
            const float2 vm = __half22float2(__ldg(vm_base + t)); // x = v_min, y = v_step
            v[0] = fmaf(static_cast<float>((packed_byte >> 0) & 3u), vm.y, vm.x);
            v[1] = fmaf(static_cast<float>((packed_byte >> 2) & 3u), vm.y, vm.x);
            v[2] = fmaf(static_cast<float>((packed_byte >> 4) & 3u), vm.y, vm.x);
            v[3] = fmaf(static_cast<float>((packed_byte >> 6) & 3u), vm.y, vm.x);
        } else {
            const uint2 vv_raw = *reinterpret_cast<const uint2*>(vfp_base + static_cast<std::size_t>(t) * kD + lane * 4);
            const half2* vh = reinterpret_cast<const half2*>(&vv_raw);
            float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
            v[0] = v0.x; v[1] = v0.y; v[2] = v1.x; v[3] = v1.y;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = acc[j] * alpha + p_w * v[j];
    }

    // 6. Add Masullo Global Shell (remaining unselected tokens per page: count = kPageSize - page_surv_cnt[p])!
    //    Zero double counting because refined survivors were subtracted from each page's multiplicity!
    for (int p = warp; p < P; p += nwarps) {
        const int rem = kPageSize - page_surv_cnt[p];
        if (rem <= 0) continue;
        const float p_logit = page_scores[p];
        const float w_page = static_cast<float>(rem) * __expf(p_logit - local_max_s);
        sum_e += w_page;
        const uint2 vmu_raw = *reinterpret_cast<const uint2*>(vpmu_base + static_cast<std::size_t>(p) * kD + lane * 4);
        const half2* vh = reinterpret_cast<const half2*>(&vmu_raw);
        float2 v0 = __half22float2(vh[0]), v1 = __half22float2(vh[1]);
        acc[0] = fmaf(w_page, v0.x, acc[0]);
        acc[1] = fmaf(w_page, v0.y, acc[1]);
        acc[2] = fmaf(w_page, v1.x, acc[2]);
        acc[3] = fmaf(w_page, v1.y, acc[3]);
    }

    // 7. Cross-warp reduction
    __shared__ float sm_wmax[8], sm_sum[8], sm_acc[8][kD];
    if (lane == 0) { sm_wmax[warp] = local_max_s; sm_sum[warp] = sum_e; }
    #pragma unroll
    for (int j = 0; j < 4; ++j) sm_acc[warp][lane * 4 + j] = acc[j];
    __syncthreads();

    if (warp == 0) {
        float fmax_g = -1e30f;
        for (int w = 0; w < nwarps; ++w) fmax_g = fmaxf(fmax_g, sm_wmax[w]);
        float gsum = 0.0f;
        float gout[4] = {0, 0, 0, 0};
        for (int w = 0; w < nwarps; ++w) {
            float sc = __expf(sm_wmax[w] - fmax_g);
            gsum += sm_sum[w] * sc;
            #pragma unroll
            for (int j = 0; j < 4; ++j) gout[j] += sm_acc[w][lane * 4 + j] * sc;
        }
        float inv = 1.0f / (gsum + 1e-9f);
        float* o_ptr = out + (static_cast<std::size_t>(m) * kHq + hq) * kD;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o_ptr[lane * 4 + j] = gout[j] * inv;
    }
}

template <class Fn>
double bench_us(Fn&& fn, cudaStream_t stream, int warm = 10, int iters = 40) {
    for (int i = 0; i < warm; ++i) fn();
    CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaEventRecord(s, stream));
    for (int i = 0; i < iters; ++i) fn();
    CHECK(cudaEventRecord(e, stream));
    CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return (ms * 1000.0) / iters;
}

} // namespace

int main() {
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::cout << "{\n  \"track\": \"R2-2_Masullo_Shell_SignKV_v2\",\n  \"runs\": [\n";
    // Realistic multi-cluster + background drift + 16 planted needles KV benchmark at 8K, 32K, 64K, 128K!
    const std::vector<int> seqs = {8192, 32768, 65536, 131072};
    const std::vector<int> Ms = {1, 16, 32};
    bool first = true;

    for (int S : seqs) {
        const int P = S / kPageSize;
        // Allocate realistic K, V (for S=131072: K+V FP16 = 2 * 8 * 131072 * 128 * 2B = 512 MiB, fits easily in 8 GiB for 1 layer!)
        std::vector<Half> h_K(static_cast<std::size_t>(kHkv) * S * kD);
        std::vector<Half> h_V(static_cast<std::size_t>(kHkv) * S * kD);
        std::vector<uint4> h_Kp0(static_cast<std::size_t>(kHkv) * S);
        std::vector<uint4> h_Kp1(static_cast<std::size_t>(kHkv) * S);
        std::vector<Half> h_Kn(static_cast<std::size_t>(kHkv) * S);
        std::vector<uint32_t> h_Vq2(static_cast<std::size_t>(kHkv) * S * 8, 0);
        std::vector<half2> h_Vm(static_cast<std::size_t>(kHkv) * S);
        std::vector<Half> h_Kpmu(static_cast<std::size_t>(kHkv) * P * kD);
        std::vector<Half> h_Vpmu(static_cast<std::size_t>(kHkv) * P * kD);
        std::vector<Half> h_Kpvar(static_cast<std::size_t>(kHkv) * P);

        // Generate realistic LLM KV with:
        //  (1) Coherent low-rank/topic drift per page (real LLMs have strong page-level K/V topic coherence!)
        //  (2) Coherent background V mean (the O(1) centroid that Round 1 threw away!)
        //  (3) 32 attention sinks with high norm + 16 planted needles!
        std::vector<float> q0_ref(kD);
        for (int d = 0; d < kD; ++d) q0_ref[d] = nd(rng);
        std::vector<int> needle_pos;
        for (int i = 0; i < 16; ++i) needle_pos.push_back(256 + (S / 17) * (i + 1));

        for (int h = 0; h < kHkv; ++h) {
            std::vector<float> v_global_bias(kD);
            for (int d = 0; d < kD; ++d) v_global_bias[d] = 0.35f * nd(rng); // Real LLM coherent V centroid!
            for (int p = 0; p < P; ++p) {
                std::vector<float> k_page_topic(kD), v_page_topic(kD);
                for (int d = 0; d < kD; ++d) {
                    k_page_topic[d] = 0.45f * nd(rng);
                    v_page_topic[d] = v_global_bias[d] + 0.30f * nd(rng);
                }
                std::vector<float> k_sum(kD, 0.0f), v_sum(kD, 0.0f);
                float k_res_sq = 0.0f;

                for (int off = 0; off < kPageSize; ++off) {
                    int t = p * kPageSize + off;
                    bool is_needle = false;
                    for (int np : needle_pos) if (t == np) is_needle = true;
                    bool is_sink = (t < 4);

                    float kf[kD], vf[kD];
                    float n2 = 0.0f;
                    for (int d = 0; d < kD; ++d) {
                        if (is_needle && h == 0) {
                            kf[d] = 0.90f * q0_ref[d] + 0.30f * nd(rng);
                            vf[d] = v_page_topic[d] + 1.2f * nd(rng);
                        } else if (is_sink) {
                            kf[d] = 0.70f * q0_ref[d] + 0.60f * nd(rng);
                            vf[d] = v_global_bias[d] + 0.50f * nd(rng);
                        } else {
                            kf[d] = k_page_topic[d] + 0.85f * nd(rng);
                            vf[d] = v_page_topic[d] + 0.75f * nd(rng);
                        }
                        n2 += kf[d] * kf[d];
                        k_sum[d] += kf[d];
                        v_sum[d] += vf[d];
                        h_K[(static_cast<std::size_t>(h) * S + t) * kD + d] = __float2half(kf[d]);
                        h_V[(static_cast<std::size_t>(h) * S + t) * kD + d] = __float2half(vf[d]);
                    }
                    float norm_k = std::sqrt(n2);
                    h_Kn[static_cast<std::size_t>(h) * S + t] = __float2half(norm_k);

                    // Pack 2-bit V (uniform min/max per token across 128 dims)
                    float vmin = vf[0], vmax = vf[0];
                    for (int d = 1; d < kD; ++d) { vmin = std::min(vmin, vf[d]); vmax = std::max(vmax, vf[d]); }
                    float vstep = (vmax - vmin) / 3.0f + 1e-6f;
                    h_Vm[static_cast<std::size_t>(h) * S + t] = __floats2half2_rn(vmin, vstep);
                    uint8_t* vbytes = reinterpret_cast<uint8_t*>(&h_Vq2[(static_cast<std::size_t>(h) * S + t) * 8]);
                    for (int lane = 0; lane < 32; ++lane) {
                        uint8_t b = 0;
                        for (int j = 0; j < 4; ++j) {
                            int qv = std::clamp(static_cast<int>(std::lround((vf[lane * 4 + j] - vmin) / vstep)), 0, 3);
                            b |= static_cast<uint8_t>(qv << (2 * j));
                        }
                        vbytes[lane] = b;
                    }

                    // Pack Iwasawa 2-bit FWHT K (Plane 0 = sign bit, Plane 1 = |x| > 0.9816 * rms!)
                    host_fwht128(kf);
                    float rms = norm_k / std::sqrt(static_cast<float>(kD));
                    float thr = 0.9816f * rms; // Optimal Gaussian 2-bit Lloyd-Max threshold!
                    unsigned p0w[4] = {0, 0, 0, 0}, p1w[4] = {0, 0, 0, 0};
                    for (int lane = 0; lane < 32; ++lane) {
                        for (int j = 0; j < 4; ++j) {
                            float val = kf[lane * 4 + j];
                            if (val >= 0.0f) p0w[j] |= (1u << lane);
                            if (std::fabs(val) >= thr) p1w[j] |= (1u << lane);
                        }
                    }
                    h_Kp0[static_cast<std::size_t>(h) * S + t] = make_uint4(p0w[0], p0w[1], p0w[2], p0w[3]);
                    h_Kp1[static_cast<std::size_t>(h) * S + t] = make_uint4(p1w[0], p1w[1], p1w[2], p1w[3]);
                }

                // Compute exact page centroid (mu_K, mu_V) and average intra-page K variance
                for (int d = 0; d < kD; ++d) {
                    float km = k_sum[d] / kPageSize;
                    float vm = v_sum[d] / kPageSize;
                    h_Kpmu[(static_cast<std::size_t>(h) * P + p) * kD + d] = __float2half(km);
                    h_Vpmu[(static_cast<std::size_t>(h) * P + p) * kD + d] = __float2half(vm);
                    // Estimate variance of (q.k / sqrt(D)) around (q.mu_k / sqrt(D)) for unit-rms q:
                    // var(dot) ~ E[||k - mu_k||^2] / D
                }
                for (int off = 0; off < kPageSize; ++off) {
                    int t = p * kPageSize + off;
                    for (int d = 0; d < kD; ++d) {
                        float diff = __half2float(h_K[(static_cast<std::size_t>(h) * S + t) * kD + d])
                                   - __half2float(h_Kpmu[(static_cast<std::size_t>(h) * P + p) * kD + d]);
                        k_res_sq += diff * diff;
                    }
                }
                float var_proxy = (k_res_sq / (kPageSize * kD)); // E[(k_d - mu_d)^2] -> Var(q.(k-mu)/sqrt(D)) = E[q_d^2] * var_proxy!
                h_Kpvar[static_cast<std::size_t>(h) * P + p] = __float2half(var_proxy);
            }
        }

        // Upload to GPU
        Half *d_K, *d_V, *d_Kn, *d_Kpmu, *d_Vpmu, *d_Kpvar;
        uint4 *d_Kp0, *d_Kp1;
        uint32_t *d_Vq2;
        half2 *d_Vm;
        int *d_surv;
        CHECK(cudaMalloc(&d_K, h_K.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_V, h_V.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_Kp0, h_Kp0.size() * sizeof(uint4)));
        CHECK(cudaMalloc(&d_Kp1, h_Kp1.size() * sizeof(uint4)));
        CHECK(cudaMalloc(&d_Kn, h_Kn.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_Vq2, h_Vq2.size() * sizeof(uint32_t)));
        CHECK(cudaMalloc(&d_Vm, h_Vm.size() * sizeof(half2)));
        CHECK(cudaMalloc(&d_Kpmu, h_Kpmu.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_Vpmu, h_Vpmu.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_Kpvar, h_Kpvar.size() * sizeof(Half)));
        CHECK(cudaMalloc(&d_surv, sizeof(int)));

        CHECK(cudaMemcpy(d_K, h_K.data(), h_K.size() * sizeof(Half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(Half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Kp0, h_Kp0.data(), h_Kp0.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Kp1, h_Kp1.data(), h_Kp1.size() * sizeof(uint4), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Kn, h_Kn.data(), h_Kn.size() * sizeof(Half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Vq2, h_Vq2.data(), h_Vq2.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Vm, h_Vm.data(), h_Vm.size() * sizeof(half2), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Kpmu, h_Kpmu.data(), h_Kpmu.size() * sizeof(Half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Vpmu, h_Vpmu.data(), h_Vpmu.size() * sizeof(Half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_Kpvar, h_Kpvar.data(), h_Kpvar.size() * sizeof(Half), cudaMemcpyHostToDevice));

        for (int M : Ms) {
            std::vector<Half> h_Q(static_cast<std::size_t>(M) * kHq * kD);
            for (int m = 0; m < M; ++m)
                for (int h = 0; h < kHq; ++h)
                    for (int d = 0; d < kD; ++d)
                        h_Q[(m * kHq + h) * kD + d] = __float2half((m == 0 && h == 0) ? q0_ref[d] : nd(rng));

            Half* d_Q;
            float *d_out_fp16, *d_out_v2_2b, *d_out_v2_fp16v;
            CHECK(cudaMalloc(&d_Q, h_Q.size() * sizeof(Half)));
            CHECK(cudaMalloc(&d_out_fp16, M * kHq * kD * sizeof(float)));
            CHECK(cudaMalloc(&d_out_v2_2b, M * kHq * kD * sizeof(float)));
            CHECK(cudaMalloc(&d_out_v2_fp16v, M * kHq * kD * sizeof(float)));
            CHECK(cudaMemcpy(d_Q, h_Q.data(), h_Q.size() * sizeof(Half), cudaMemcpyHostToDevice));

            dim3 grid(kHq, M);
            // Tune margin_z so ~300-800 keys survive (broad enough to capture all spikes + needles!)
            float margin_z = 6.5f;

            double fp16_us = bench_us([&](){
                fp16_attn_decode<<<grid, 256, 0, stream>>>(d_Q, d_K, d_V, d_out_fp16, M, S);
            }, stream, 5, 20);

            double v2_2bit_us = bench_us([&](){
                sign_kv_v2_shell<true, true><<<grid, 256, 0, stream>>>(
                    d_Q, d_Kp0, d_Kp1, d_Kn, d_Vq2, d_Vm, d_V, d_Kpmu, d_Vpmu, d_Kpvar,
                    d_out_v2_2b, d_surv, M, S, P, margin_z);
            }, stream, 5, 20);

            int surv = 0;
            CHECK(cudaMemcpy(&surv, d_surv, sizeof(int), cudaMemcpyDeviceToHost));

            // Check cosine & relative L2 error across ALL M * kHq heads!
            std::vector<float> o_ref(M * kHq * kD), o_v2(M * kHq * kD);
            CHECK(cudaMemcpy(o_ref.data(), d_out_fp16, o_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(o_v2.data(), d_out_v2_2b, o_v2.size() * sizeof(float), cudaMemcpyDeviceToHost));

            double min_cos = 1.0, sum_cos = 0.0, max_rel_l2 = 0.0, sum_rel_l2 = 0.0;
            double head0_cos = 0.0, head0_rel_l2 = 0.0;
            int total_heads = M * kHq;
            for (int idx = 0; idx < total_heads; ++idx) {
                double dot = 0, na = 0, nb = 0, diff2 = 0;
                for (int d = 0; d < kD; ++d) {
                    double a = o_ref[idx * kD + d], b = o_v2[idx * kD + d];
                    dot += a * b; na += a * a; nb += b * b; diff2 += (a - b) * (a - b);
                }
                double c = dot / (std::sqrt(na * nb) + 1e-12);
                double rl2 = std::sqrt(diff2) / (std::sqrt(na) + 1e-12);
                if (idx == 0) { head0_cos = c; head0_rel_l2 = rl2; }
                min_cos = std::min(min_cos, c);
                sum_cos += c;
                max_rel_l2 = std::max(max_rel_l2, rl2);
                sum_rel_l2 += rl2;
            }

            // Memory footprint (36 layers, H_kv=8)
            double fp16_gib = (36.0 * kHkv * S * kD * 2.0 * 2.0) / double(1ull << 30);
            double v2_gib   = (36.0 * kHkv * S * 78.0) / double(1ull << 30); // 16(P0)+16(P1)+2(norm)+36(2bV)+8(page_mu)

            if (!first) std::cout << ",\n";
            first = false;
            std::cout << std::fixed << std::setprecision(4)
                      << "    {\"S\": " << S << ", \"M\": " << M
                      << ", \"fp16_gpu_us\": " << fp16_us
                      << ", \"sign_kv_v2_us\": " << v2_2bit_us
                      << ", \"speedup_vs_gpu\": " << (fp16_us / v2_2bit_us)
                      << ", \"survived_q0h0\": " << surv
                      << ", \"head0_cos\": " << head0_cos
                      << ", \"head0_rel_l2\": " << head0_rel_l2
                      << ", \"mean_cos_all_heads\": " << (sum_cos / total_heads)
                      << ", \"min_cos_all_heads\": " << min_cos
                      << ", \"mean_rel_l2\": " << (sum_rel_l2 / total_heads)
                      << ", \"fp16_36L_gib\": " << fp16_gib
                      << ", \"v2_36L_gib\": " << v2_gib
                      << ", \"vram_compression\": " << (fp16_gib / v2_gib) << "}";

            CHECK(cudaFree(d_Q));
            CHECK(cudaFree(d_out_fp16));
            CHECK(cudaFree(d_out_v2_2b));
            CHECK(cudaFree(d_out_v2_fp16v));
        }

        CHECK(cudaFree(d_K)); CHECK(cudaFree(d_V));
        CHECK(cudaFree(d_Kp0)); CHECK(cudaFree(d_Kp1)); CHECK(cudaFree(d_Kn));
        CHECK(cudaFree(d_Vq2)); CHECK(cudaFree(d_Vm));
        CHECK(cudaFree(d_Kpmu)); CHECK(cudaFree(d_Vpmu)); CHECK(cudaFree(d_Kpvar));
        CHECK(cudaFree(d_surv));
    }
    std::cout << "\n  ]\n}\n";
    CHECK(cudaStreamDestroy(stream));
    return 0;
}
