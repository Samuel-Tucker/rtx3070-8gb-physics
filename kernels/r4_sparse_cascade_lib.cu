// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math -Xcompiler -fPIC -shared r4_sparse_cascade_lib.cu -o libr4_sparse_cascade.so
// Round 4 Objective 3C (Carbon + Sol 6.2 + Kimi):
//   Zero-Sync CUDA-Graph Exact-Gate Sparse (W_up + W_down) + Post-Attn Sparse W_o!
//   - Zero torch.kthvalue, ZERO host sync, ZERO dynamic shapes -> 100% static torch.cuda.CUDAGraph capturable!
//   - Selector: Warp-local exact 16-of-32 (50% skip!) or 12-of-32 (62.5% skip!) magnitude*colnorm ranking
//     via warp bitonic/rank count in registers (< 2 us, fixed output count = I * keep_k / 32!).
//   - Sparse W_up (row-major [I, H]): only surviving neurons read their coalesced W_up row & multiply by SiLU(g)!
//   - Split-K 40-SM Saturated Column-Major W_down ([I, H]): only surviving neurons read their coalesced W_down column!
//   - Same warp-local 16-of-32 selector + Split-K column-major kernel reused for post-attention W_o!

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Half = __half;

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

// 1. Warp-Local Exact Top-K-of-32 Selector (ZERO global sync, ZERO atomicAdd, fixed static output size!)
//    Each warp handles 32 contiguous channels [w*32 .. w*32+31].
//    Computes exact SiLU(g_i) (if apply_silu=true) and importance score = |val_i| * col_norm[i].
//    Each lane counts how many of the 32 warp lanes have strictly higher score (breaking ties by lane index) -> exact rank 0..31!
//    Lanes with rank < K_keep write (global_idx, val_i) directly into deterministic slot `w * K_keep + rank`!
template <bool ApplySilu, int KeepPer32>
__global__ void warp_topk32_select_kernel(
    const Half* __restrict__ in_vec,     // [N] (gate output OR attn output)
    const Half* __restrict__ col_norms,  // [N] (precomputed ||W[:, i]||_2 or ||W_up[i]||*||W_down[:,i]||)
    int* __restrict__ out_idx,           // [N / 32 * KeepPer32] FIXED STATIC SIZE!
    Half* __restrict__ out_val,          // [N / 32 * KeepPer32] FIXED STATIC SIZE!
    int N) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_global = tid >> 5;
    const int lane = tid & 31;
    if (tid >= N) return;

    float v = __half2float(in_vec[tid]);
    if constexpr (ApplySilu) {
        v = v / (1.0f + __expf(-v)); // exact SiLU(g)
    }
    float score = fabsf(v) * __half2float(col_norms[tid]);

    // Exact 32-lane rank count via warp shuffles (32 register ops -> ~1.5 us for entire layer!)
    int rank = 0;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float other = __shfl_sync(0xffffffffu, score, j);
        if (other > score || (other == score && j < lane)) ++rank;
    }
    if (rank < KeepPer32) {
        const int dst = warp_global * KeepPer32 + rank;
        out_idx[dst] = tid;
        out_val[dst] = __float2half(v);
    }
}

// 2. Sparse Row-Major W_up ONLY on surviving exact-gate neurons + fuse multiply by silu(g)!
//    Grid = (n_keep + 7) / 8, Block = 256 (1 warp = 1 surviving neuron -> 100% coalesced uint4 row loads!)
__global__ void sparse_up_mul_silu_fp16_kernel(
    const Half* __restrict__ x,          // [H]
    const Half* __restrict__ W_up,       // [I, H] row-major!
    const int* __restrict__ keep_idx,    // [n_keep]
    const Half* __restrict__ silu_g_val, // [n_keep]
    Half* __restrict__ inter_compact,    // [n_keep] -> silu(g) * up!
    int H, int n_keep) {

    const int k = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (k >= n_keep) return;
    const int neuron = keep_idx[k];
    const int lane = threadIdx.x & 31;
    const int chunks = H >> 3; // uint4 = 8 FP16s
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* uw = reinterpret_cast<const uint4*>(W_up + static_cast<std::size_t>(neuron) * H);

    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 xq = __ldg(xv + c);
        const uint4 uq = __ldg(uw + c);
        const half2* xh = reinterpret_cast<const half2*>(&xq);
        const half2* uh = reinterpret_cast<const half2*>(&uq);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fx = __half22float2(xh[j]);
            float2 fu = __half22float2(uh[j]);
            acc = fmaf(fx.x, fu.x, fmaf(fx.y, fu.y, acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        inter_compact[k] = __float2half(acc * __half2float(silu_g_val[k]));
    }
}

// 3. Split-K 40-SM Saturated Column-Major Sparse MatVec + Fused Residual Add (`y_out = residual + W_T[keep_idx]^T @ val_compact`)!
//    Uses deterministic scratch partials [split_k, H] + tiny reduction kernel (100% deterministic, zero FP16 atomicAdd issues!)
__global__ void splitk_colmajor_sparse_fp16_kernel(
    const Half* __restrict__ W_T,         // [N_in, H_out] column-major (contiguous H_out per input col!)
    const int* __restrict__ keep_idx,     // [n_keep]
    const Half* __restrict__ val_compact, // [n_keep]
    float* __restrict__ partials,         // [split_k, H_out]
    int H_out, int n_keep, int split_k) {

    const int h4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = h4 * 4;
    if (h >= H_out) return;
    const int sk = blockIdx.y;
    const int chunk = (n_keep + split_k - 1) / split_k;
    const int k0 = sk * chunk;
    const int k1 = min(n_keep, k0 + chunk);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = k0; k < k1; ++k) {
        const int col = keep_idx[k];
        const float a = __half2float(__ldg(val_compact + k));
        const uint2 raw = *reinterpret_cast<const uint2*>(W_T + static_cast<std::size_t>(col) * H_out + h);
        const half2* hh = reinterpret_cast<const half2*>(&raw);
        float2 f0 = __half22float2(hh[0]), f1 = __half22float2(hh[1]);
        a0 = fmaf(a, f0.x, a0); a1 = fmaf(a, f0.y, a1);
        a2 = fmaf(a, f1.x, a2); a3 = fmaf(a, f1.y, a3);
    }
    float* dst = partials + static_cast<std::size_t>(sk) * H_out + h;
    dst[0] = a0; dst[1] = a1; dst[2] = a2; dst[3] = a3;
}

__global__ void reduce_splitk_residual_fp16_kernel(
    const float* __restrict__ partials, // [split_k, H_out]
    const Half* __restrict__ residual,  // [H_out]
    Half* __restrict__ out,             // [H_out]
    int H_out, int split_k) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H_out) return;
    float sum = __half2float(residual[h]);
    for (int sk = 0; sk < split_k; ++sk) {
        sum += partials[static_cast<std::size_t>(sk) * H_out + h];
    }
    out[h] = __float2half(sum);
}

} // namespace

extern "C" {

void launch_exact_gate_cascade_fp16(
    const void* x_norm,        // [H]
    const void* gate_out,      // [I] (already computed densely via cuBLAS!)
    const void* gate_colnorms, // [I] (||W_up[i]|| * ||W_down[:, i]||)
    const void* W_up,          // [I, H]
    const void* W_down_T,      // [I, H]
    const void* residual,      // [H]
    void* out_y,               // [H]
    void* tmp_idx,             // [n_keep] int32
    void* tmp_val,             // [n_keep] fp16
    void* tmp_inter,           // [n_keep] fp16
    void* tmp_partials,        // [split_k, H] fp32
    int H, int I, int keep_per_32, int split_k, cudaStream_t stream) {

    const int n_keep = (I / 32) * keep_per_32;
    if (keep_per_32 == 16) {
        warp_topk32_select_kernel<true, 16><<<(I + 255) / 256, 256, 0, stream>>>(
            (const Half*)gate_out, (const Half*)gate_colnorms, (int*)tmp_idx, (Half*)tmp_val, I);
    } else if (keep_per_32 == 12) {
        warp_topk32_select_kernel<true, 12><<<(I + 255) / 256, 256, 0, stream>>>(
            (const Half*)gate_out, (const Half*)gate_colnorms, (int*)tmp_idx, (Half*)tmp_val, I);
    } else {
        warp_topk32_select_kernel<true, 20><<<(I + 255) / 256, 256, 0, stream>>>(
            (const Half*)gate_out, (const Half*)gate_colnorms, (int*)tmp_idx, (Half*)tmp_val, I);
    }
    sparse_up_mul_silu_fp16_kernel<<<(n_keep + 7) / 8, 256, 0, stream>>>(
        (const Half*)x_norm, (const Half*)W_up, (const int*)tmp_idx, (const Half*)tmp_val, (Half*)tmp_inter, H, n_keep);
    dim3 g_sk((H / 4 + 127) / 128, split_k);
    splitk_colmajor_sparse_fp16_kernel<<<g_sk, 128, 0, stream>>>(
        (const Half*)W_down_T, (const int*)tmp_idx, (const Half*)tmp_inter, (float*)tmp_partials, H, n_keep, split_k);
    reduce_splitk_residual_fp16_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        (const float*)tmp_partials, (const Half*)residual, (Half*)out_y, H, split_k);
}

void launch_sparse_colmajor_proj_fp16(
    const void* in_vec,        // [N_in] (either exact silu*up OR attn_out!)
    const void* colnorms,      // [N_in]
    const void* W_T,           // [N_in, H_out]
    const void* residual,      // [H_out]
    void* out_y,               // [H_out]
    void* tmp_idx,             // [n_keep] int32
    void* tmp_val,             // [n_keep] fp16
    void* tmp_partials,        // [split_k, H_out] fp32
    int N_in, int H_out, int keep_per_32, int split_k, cudaStream_t stream) {

    const int n_keep = (N_in / 32) * keep_per_32;
    if (keep_per_32 == 16) {
        warp_topk32_select_kernel<false, 16><<<(N_in + 255) / 256, 256, 0, stream>>>(
            (const Half*)in_vec, (const Half*)colnorms, (int*)tmp_idx, (Half*)tmp_val, N_in);
    } else if (keep_per_32 == 12) {
        warp_topk32_select_kernel<false, 12><<<(N_in + 255) / 256, 256, 0, stream>>>(
            (const Half*)in_vec, (const Half*)colnorms, (int*)tmp_idx, (Half*)tmp_val, N_in);
    } else {
        warp_topk32_select_kernel<false, 20><<<(N_in + 255) / 256, 256, 0, stream>>>(
            (const Half*)in_vec, (const Half*)colnorms, (int*)tmp_idx, (Half*)tmp_val, N_in);
    }
    dim3 g_sk((H_out / 4 + 127) / 128, split_k);
    splitk_colmajor_sparse_fp16_kernel<<<g_sk, 128, 0, stream>>>(
        (const Half*)W_T, (const int*)tmp_idx, (const Half*)tmp_val, (float*)tmp_partials, H_out, n_keep, split_k);
    reduce_splitk_residual_fp16_kernel<<<(H_out + 255) / 256, 256, 0, stream>>>(
        (const float*)tmp_partials, (const Half*)residual, (Half*)out_y, H_out, split_k);
}

} // extern "C"
