// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math -Xcompiler -fPIC -shared qwen4b_gguf_ops.cu -o libqwen4b_gguf.so
// Exact ggml Q4_K (type 12, 144 B/256) and Q6_K (type 14, 210 B/256) CUDA GEMV (M=1) & Dequant-to-FP16 (for chunked prefill!)
// Enables running REAL 36-layer Qwen3-4B-Instruct-2507-Q4_K_M (native 256K context, 2.33 GiB GGUF!) directly with
// Masullo-Shell Sign-KV v3 (0.46 GiB at 64K!) inside ~3.1 GiB total GPU VRAM on an 8 GB RTX 3070!

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using Half = __half;

#pragma pack(push, 1)
struct block_q4_K {
    Half d;
    Half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    Half    d;
};
#pragma pack(pop)

static_assert(sizeof(block_q4_K) == 144, "Q4_K block size must be 144");
static_assert(sizeof(block_q6_K) == 210, "Q6_K block size must be 210");

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

// Exact Q4_K Dequantization to FP16 [N, K] (for chunked prefill & embedding lookup!)
// Grid = (N, K / 256), Block = 32 (1 warp per 256-element Q4_K block -> 8 elements per lane!)
__global__ void dequant_q4k_kernel(const block_q4_K* __restrict__ W, Half* __restrict__ out, int N, int nb) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (row >= N || b >= nb) return;

    const block_q4_K* blk = W + static_cast<std::size_t>(row) * nb + b;
    const float d = __half2float(blk->d);
    const float min_v = __half2float(blk->dmin);
    Half* dst = out + (static_cast<std::size_t>(row) * nb + b) * 256;

    // 4 chunks of 64 weights (32 low-nibbles + 32 high-nibbles)
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(2 * c,     blk->scales, sc0, m0);
        get_scale_min_k4(2 * c + 1, blk->scales, sc1, m1);
        const float d1 = d * sc0, dm1 = min_v * m0;
        const float d2 = d * sc1, dm2 = min_v * m1;
        const uint8_t q = blk->qs[c * 32 + lane];
        dst[c * 64 + lane]      = __float2half(d1 * (q & 0xF) - dm1);
        dst[c * 64 + 32 + lane] = __float2half(d2 * (q >> 4)  - dm2);
    }
}

// Fast Coalesced Q4_K GEMV (y[N] = W_q4k[N, K] @ x[K], 1 warp per output row!)
// Grid = (N + 7) / 8, Block = 256 (8 warps/block)
__global__ void gemv_q4k_kernel(const block_q4_K* __restrict__ W, const Half* __restrict__ x,
                                Half* __restrict__ y, int N, int nb) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;

    const block_q4_K* row_w = W + static_cast<std::size_t>(row) * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q4_K* blk = row_w + b;
        const float d = __half2float(__ldg(&blk->d));
        const float min_v = __half2float(__ldg(&blk->dmin));
        const Half* xb = x + b * 256;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(2 * c,     blk->scales, sc0, m0);
            get_scale_min_k4(2 * c + 1, blk->scales, sc1, m1);
            const uint8_t q = __ldg(&blk->qs[c * 32 + lane]);
            const float x0 = __half2float(__ldg(xb + c * 64 + lane));
            const float x1 = __half2float(__ldg(xb + c * 64 + 32 + lane));
            acc = fmaf(d * sc0 * (q & 0xF) - min_v * m0, x0, acc);
            acc = fmaf(d * sc1 * (q >> 4)  - min_v * m1, x1, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc);
}

// Exact Q6_K Dequantization to FP16 [N, K]
// Grid = (N, K / 256), Block = 32 (1 warp per 256-element Q6_K block)
__global__ void dequant_q6k_kernel(const block_q6_K* __restrict__ W, Half* __restrict__ out, int N, int nb) {
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (row >= N || b >= nb) return;

    const block_q6_K* blk = W + static_cast<std::size_t>(row) * nb + b;
    const float d = __half2float(blk->d);
    Half* dst = out + (static_cast<std::size_t>(row) * nb + b) * 256;

    // Two 128-element halves (n = 0, 128)
    #pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        const uint8_t* ql = blk->ql + half_idx * 64;
        const uint8_t* qh = blk->qh + half_idx * 32;
        const int8_t*  sc = blk->scales + half_idx * 8;
        Half* yd = dst + half_idx * 128;

        const int is = lane >> 4; // 0 for lanes 0..15, 1 for lanes 16..31
        const int q1 = static_cast<int>((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;

        yd[lane +  0] = __float2half(d * sc[is + 0] * q1);
        yd[lane + 32] = __float2half(d * sc[is + 2] * q2);
        yd[lane + 64] = __float2half(d * sc[is + 4] * q3);
        yd[lane + 96] = __float2half(d * sc[is + 6] * q4);
    }
}

// Fast Coalesced Q6_K GEMV (y[N] = W_q6k[N, K] @ x[K], 1 warp per output row!)
__global__ void gemv_q6k_kernel(const block_q6_K* __restrict__ W, const Half* __restrict__ x,
                                Half* __restrict__ y, int N, int nb) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= N) return;

    const block_q6_K* row_w = W + static_cast<std::size_t>(row) * nb;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q6_K* blk = row_w + b;
        const float d = __half2float(__ldg(&blk->d));
        const Half* xb = x + b * 256;
        #pragma unroll
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const uint8_t* ql = blk->ql + half_idx * 64;
            const uint8_t* qh = blk->qh + half_idx * 32;
            const int8_t*  sc = blk->scales + half_idx * 8;
            const Half* xh = xb + half_idx * 128;
            const int is = lane >> 4;
            const uint8_t ql0 = __ldg(&ql[lane]);
            const uint8_t ql1 = __ldg(&ql[lane + 32]);
            const uint8_t qhv = __ldg(&qh[lane]);
            const int q1 = static_cast<int>((ql0 & 0xF) | (((qhv >> 0) & 3) << 4)) - 32;
            const int q2 = static_cast<int>((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
            const int q3 = static_cast<int>((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32;
            const int q4 = static_cast<int>((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32;
            acc = fmaf(d * __ldg(&sc[is + 0]) * q1, __half2float(__ldg(xh + lane +  0)), acc);
            acc = fmaf(d * __ldg(&sc[is + 2]) * q2, __half2float(__ldg(xh + lane + 32)), acc);
            acc = fmaf(d * __ldg(&sc[is + 4]) * q3, __half2float(__ldg(xh + lane + 64)), acc);
            acc = fmaf(d * __ldg(&sc[is + 6]) * q4, __half2float(__ldg(xh + lane + 96)), acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc);
}

// Fast Q4_K / Q6_K single-row embedding gather for token IDs [T] -> out[T, K] FP16
__global__ void embed_q4k_kernel(const block_q4_K* __restrict__ W, const int64_t* __restrict__ ids,
                                 Half* __restrict__ out, int T, int nb) {
    const int ti = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (ti >= T || b >= nb) return;
    const int64_t row = ids[ti];
    const block_q4_K* blk = W + static_cast<std::size_t>(row) * nb + b;
    const float d = __half2float(blk->d), min_v = __half2float(blk->dmin);
    Half* dst = out + (static_cast<std::size_t>(ti) * nb + b) * 256;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(2 * c, blk->scales, sc0, m0);
        get_scale_min_k4(2 * c + 1, blk->scales, sc1, m1);
        const uint8_t q = blk->qs[c * 32 + lane];
        dst[c * 64 + lane]      = __float2half(d * sc0 * (q & 0xF) - min_v * m0);
        dst[c * 64 + 32 + lane] = __float2half(d * sc1 * (q >> 4)  - min_v * m1);
    }
}

__global__ void embed_q6k_kernel(const block_q6_K* __restrict__ W, const int64_t* __restrict__ ids,
                                 Half* __restrict__ out, int T, int nb) {
    const int ti = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (ti >= T || b >= nb) return;
    const int64_t row = ids[ti];
    const block_q6_K* blk = W + static_cast<std::size_t>(row) * nb + b;
    const float d = __half2float(blk->d);
    Half* dst = out + (static_cast<std::size_t>(ti) * nb + b) * 256;
    #pragma unroll
    for (int half_idx = 0; half_idx < 2; ++half_idx) {
        const uint8_t* ql = blk->ql + half_idx * 64;
        const uint8_t* qh = blk->qh + half_idx * 32;
        const int8_t*  sc = blk->scales + half_idx * 8;
        Half* yd = dst + half_idx * 128;
        const int is = lane >> 4;
        const int q1 = static_cast<int>((ql[lane]      & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql[lane]      >> 4)  | (((qh[lane] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql[lane + 32] >> 4)  | (((qh[lane] >> 6) & 3) << 4)) - 32;
        yd[lane +  0] = __float2half(d * sc[is + 0] * q1);
        yd[lane + 32] = __float2half(d * sc[is + 2] * q2);
        yd[lane + 64] = __float2half(d * sc[is + 4] * q3);
        yd[lane + 96] = __float2half(d * sc[is + 6] * q4);
    }
}

} // namespace

extern "C" {

void launch_dequant_q4k(const void* W, void* out, int N, int K) {
    dim3 g(N, K / 256);
    dequant_q4k_kernel<<<g, 32>>>((const block_q4_K*)W, (Half*)out, N, K / 256);
}
void launch_gemv_q4k(const void* W, const void* x, void* y, int N, int K) {
    gemv_q4k_kernel<<<(N + 7) / 8, 256>>>((const block_q4_K*)W, (const Half*)x, (Half*)y, N, K / 256);
}
void launch_dequant_q6k(const void* W, void* out, int N, int K) {
    dim3 g(N, K / 256);
    dequant_q6k_kernel<<<g, 32>>>((const block_q6_K*)W, (Half*)out, N, K / 256);
}
void launch_gemv_q6k(const void* W, const void* x, void* y, int N, int K) {
    gemv_q6k_kernel<<<(N + 7) / 8, 256>>>((const block_q6_K*)W, (const Half*)x, (Half*)y, N, K / 256);
}
void launch_embed_q4k(const void* W, const void* ids, void* out, int T, int K) {
    dim3 g(T, K / 256);
    embed_q4k_kernel<<<g, 32>>>((const block_q4_K*)W, (const int64_t*)ids, (Half*)out, T, K / 256);
}
void launch_embed_q6k(const void* W, const void* ids, void* out, int T, int K) {
    dim3 g(T, K / 256);
    embed_q6k_kernel<<<g, 32>>>((const block_q6_K*)W, (const int64_t*)ids, (Half*)out, T, K / 256);
}

} // extern "C"
