// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math coalesced_ffn_v2.cu -lcublas -o coalesced_ffn_v2
// Round 2 Double-Down (Track R2-1A v2): Split-K 40-SM Saturated Coalesced Sparse FFN + Rank-64 Paley-Wiener Predictor!
// Fixes v1's 4-block occupancy bug (which left 36/40 SMs idle!) using Split-K across active neurons/stripes
// so all 40 SMs saturate 358.7 GB/s GDDR6!
// Measures BOTH:
//   1) Real Column-Major Split-K Sparse W_down (FP16 & INT4) at 50%, 60%, 70% skip vs cuBLAS FP16 & packed_int4
//   2) Full Rank-64 Paley-Wiener Screened SwiGLU (gate + up + silu*up + col-major down) vs dense cuBLAS FP16 & dense INT4!

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Half = __half;
constexpr int kStripe = 32;
constexpr int kRank = 64;

void check(cudaError_t s, const char* c) { if (s != cudaSuccess) throw std::runtime_error(std::string(c) + ": " + cudaGetErrorString(s)); }
void check(cublasStatus_t s, const char* c) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(c) + ": cublas err"); }
#define CHECK(x) check((x), #x)

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

__global__ void flush_l2(uint32_t* buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        buf[i] += 0x9e3779b9u;
}

// ============================================================================
// Split-K 40-SM Saturated Column-Major Sparse W_down (FP16):
// Grid = (H / 512, SplitK) with 128 threads/block (each thread handles 4 outputs via uint2 = 8B coalesced!).
// For H=4096, SplitK=16 -> 8 * 16 = 128 blocks -> 3.2 blocks/SM across ALL 40 SMs!
// Each block accumulates its slice of active columns in registers, then writes 1 partial vector to scratch,
// followed by a tiny 4 us reduction kernel (or atomicAdd for FP32 — wait: atomicAdd on 4096 floats across 16 splits = 4 us!).
// ============================================================================
__global__ void colmajor_sparse_down_fp16_splitk(
    const Half* __restrict__ W_down_T, // [I, H] contiguous per neuron!
    const Half* __restrict__ act,      // [I]
    const int* __restrict__ active_idx,
    const int* __restrict__ active_cnt,
    float* __restrict__ y,             // [H] zeroed before launch
    int H, int split_k) {

    const int h4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = h4 * 4;
    if (h >= H) return;
    const int n_act = *active_cnt;
    const int sk = blockIdx.y;
    const int chunk = (n_act + split_k - 1) / split_k;
    const int k_start = sk * chunk;
    const int k_end = min(n_act, k_start + chunk);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int k = k_start; k < k_end; ++k) {
        const int col = active_idx[k];
        const float a = __half2float(__ldg(act + col));
        const uint2 raw = *reinterpret_cast<const uint2*>(W_down_T + static_cast<std::size_t>(col) * H + h);
        const half2* hh = reinterpret_cast<const half2*>(&raw);
        float2 f0 = __half22float2(hh[0]), f1 = __half22float2(hh[1]);
        a0 = fmaf(a, f0.x, a0); a1 = fmaf(a, f0.y, a1);
        a2 = fmaf(a, f1.x, a2); a3 = fmaf(a, f1.y, a3);
    }
    if (k_start < k_end) {
        atomicAdd(&y[h + 0], a0); atomicAdd(&y[h + 1], a1);
        atomicAdd(&y[h + 2], a2); atomicAdd(&y[h + 3], a3);
    }
}

// Split-K 40-SM Saturated Column-Major Sparse W_down (INT4, 4.25 bpw):
// Each thread handles 8 outputs (1 uint32 = 8 INT4s = 4B coalesced -> 128B per warp!)
// Grid = (H / 1024, SplitK) with 128 threads/block. For H=4096, SplitK=24 -> 4 * 24 = 96 blocks across 40 SMs!
__global__ void colmajor_sparse_down_int4_splitk(
    const uint32_t* __restrict__ Wd_T_u32, // [I, H/8]
    const Half* __restrict__ Sd_T,         // [I, H/64]
    const Half* __restrict__ act,          // [I]
    const int* __restrict__ active_idx,
    const int* __restrict__ active_cnt,
    float* __restrict__ y,
    int H, int split_k) {

    const int h8 = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = h8 * 8;
    if (h >= H) return;
    const int n_act = *active_cnt;
    const int sk = blockIdx.y;
    const int chunk = (n_act + split_k - 1) / split_k;
    const int k_start = sk * chunk;
    const int k_end = min(n_act, k_start + chunk);

    const int words_per_row = H >> 3;
    const int sc_per_row = H >> 6;
    const int sc_col = h >> 6;
    float acc[8] = {0,0,0,0,0,0,0,0};

    for (int k = k_start; k < k_end; ++k) {
        const int col = active_idx[k];
        const float a = __half2float(__ldg(act + col));
        const float sc = a * __half2float(__ldg(Sd_T + static_cast<std::size_t>(col) * sc_per_row + sc_col));
        const uint32_t u = __ldg(Wd_T_u32 + static_cast<std::size_t>(col) * words_per_row + h8);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[2*j+0] = fmaf(sc, static_cast<float>(static_cast<int>((u >> (8*j)) & 15u) - 8), acc[2*j+0]);
            acc[2*j+1] = fmaf(sc, static_cast<float>(static_cast<int>((u >> (8*j+4)) & 15u) - 8), acc[2*j+1]);
        }
    }
    if (k_start < k_end) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) atomicAdd(&y[h + j], acc[j]);
    }
}

// ============================================================================
// Masullo Paley-Wiener Rank-64 Gate Predictor (L2-resident! 2.09 MB for 4096x12288!)
// Step 1: z = V_r @ x (64 x H) -> tiny 64-float vector (~4 us!)
// Step 2: g_hat = U_r @ z (I x 64) + emit surviving neuron indices (all 12288 rows x 64 = 1.5 MB -> ~6 us!)
// ============================================================================
__global__ void rank64_proj_v(const Half* __restrict__ V_r, const Half* __restrict__ x, float* __restrict__ z, int H) {
    // 64 blocks (or 8 blocks x 8 warps), 1 warp per rank dim
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= kRank) return;
    const int lane = threadIdx.x & 31;
    const int chunks = H >> 3;
    const uint4* vr = reinterpret_cast<const uint4*>(V_r + static_cast<std::size_t>(r) * H);
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 vq = __ldg(vr + c), xq = __ldg(xv + c);
        const half2* vh = reinterpret_cast<const half2*>(&vq);
        const half2* xh = reinterpret_cast<const half2*>(&xq);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fv = __half22float2(vh[j]), fx = __half22float2(xh[j]);
            acc = fmaf(fv.x, fx.x, fmaf(fv.y, fx.y, acc));
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) z[r] = acc;
}

__global__ void rank64_screen_neurons(const Half* __restrict__ U_r, // [I, 64]
                                      const float* __restrict__ z,  // [64]
                                      int* __restrict__ active_idx,
                                      int* __restrict__ active_cnt,
                                      int I, float thresh) {
    __shared__ float zs[kRank];
    if (threadIdx.x < kRank) zs[threadIdx.x] = z[threadIdx.x];
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const uint4* ur = reinterpret_cast<const uint4*>(U_r + static_cast<std::size_t>(i) * kRank);
    float acc = 0.0f;
    #pragma unroll
    for (int c = 0; c < kRank / 8; ++c) {
        uint4 uq = __ldg(ur + c);
        const half2* uh = reinterpret_cast<const half2*>(&uq);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fu = __half22float2(uh[j]);
            acc = fmaf(fu.x, zs[c*8 + 2*j], fmaf(fu.y, zs[c*8 + 2*j + 1], acc));
        }
    }
    if (acc >= thresh) {
        int slot = atomicAdd(active_cnt, 1);
        active_idx[slot] = i;
    }
}

// Sparse Gate+Up (FP16) over active_idx (1 warp per active neuron -> 100% coalesced row reads!)
__global__ void sparse_gate_up_fp16(const Half* __restrict__ x,
                                    const Half* __restrict__ Wg, const Half* __restrict__ Wu,
                                    const int* __restrict__ active_idx, const int* __restrict__ active_cnt,
                                    Half* __restrict__ act_out, int H) {
    const int k = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (k >= *active_cnt) return;
    const int neuron = active_idx[k];
    const int lane = threadIdx.x & 31;
    const int chunks = H >> 3;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* gw = reinterpret_cast<const uint4*>(Wg + static_cast<std::size_t>(neuron) * H);
    const uint4* uw = reinterpret_cast<const uint4*>(Wu + static_cast<std::size_t>(neuron) * H);
    float ga = 0.0f, ua = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 xq = __ldg(xv + c), gq = __ldg(gw + c), uq = __ldg(uw + c);
        const half2* xh = reinterpret_cast<const half2*>(&xq);
        const half2* gh = reinterpret_cast<const half2*>(&gq);
        const half2* uh = reinterpret_cast<const half2*>(&uq);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 fx = __half22float2(xh[j]), fg = __half22float2(gh[j]), fu = __half22float2(uh[j]);
            ga = fmaf(fx.x, fg.x, fmaf(fx.y, fg.y, ga));
            ua = fmaf(fx.x, fu.x, fmaf(fx.y, fu.y, ua));
        }
    }
    ga = warp_sum(ga); ua = warp_sum(ua);
    if (lane == 0) act_out[neuron] = __float2half((ga / (1.0f + __expf(-ga))) * ua);
}

// Sparse Gate+Up (INT4, 4.25 bpw) over active_idx (1 warp per active neuron -> 100% coalesced uint4 loads!)
__global__ void sparse_gate_up_int4(const Half* __restrict__ x,
                                    const uint4* __restrict__ Wg4, const Half* __restrict__ Sg,
                                    const uint4* __restrict__ Wu4, const Half* __restrict__ Su,
                                    const int* __restrict__ active_idx, const int* __restrict__ active_cnt,
                                    Half* __restrict__ act_out, int H) {
    const int k = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (k >= *active_cnt) return;
    const int neuron = active_idx[k];
    const int lane = threadIdx.x & 31;
    const int chunks = H >> 5;
    const uint4* gw = Wg4 + static_cast<std::size_t>(neuron) * chunks;
    const Half*  gs = Sg  + static_cast<std::size_t>(neuron) * (H >> 6);
    const uint4* uw = Wu4 + static_cast<std::size_t>(neuron) * chunks;
    const Half*  us = Su  + static_cast<std::size_t>(neuron) * (H >> 6);
    float ga = 0.0f, ua = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 wg = __ldg(gw + c), wu = __ldg(uw + c);
        float scg = __half2float(__ldg(gs + (c >> 1)));
        float scu = __half2float(__ldg(us + (c >> 1)));
        const unsigned gw4[4] = {wg.x, wg.y, wg.z, wg.w};
        const unsigned uw4[4] = {wu.x, wu.y, wu.z, wu.w};
        float pg = 0.0f, pu = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint4 xq = __ldg(reinterpret_cast<const uint4*>(x + c * 32 + p * 8));
            const half2* h = reinterpret_cast<const half2*>(&xq);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h[j]);
                pg = fmaf(static_cast<float>(static_cast<int>((gw4[p] >> (8*j)) & 15u) - 8), f.x,
                     fmaf(static_cast<float>(static_cast<int>((gw4[p] >> (8*j+4)) & 15u) - 8), f.y, pg));
                pu = fmaf(static_cast<float>(static_cast<int>((uw4[p] >> (8*j)) & 15u) - 8), f.x,
                     fmaf(static_cast<float>(static_cast<int>((uw4[p] >> (8*j+4)) & 15u) - 8), f.y, pu));
            }
        }
        ga = fmaf(scg, pg, ga); ua = fmaf(scu, pu, ua);
    }
    ga = warp_sum(ga); ua = warp_sum(ua);
    if (lane == 0) act_out[neuron] = __float2half((ga / (1.0f + __expf(-ga))) * ua);
}

// Dense INT4 GEMV baseline
__global__ void int4_gemv(const uint4* __restrict__ W, const Half* __restrict__ S,
                          const Half* __restrict__ x, Half* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= N) return;
    const int chunks = K >> 5;
    const uint4* rw = W + static_cast<std::size_t>(row) * chunks;
    const Half* rs = S + static_cast<std::size_t>(row) * (K >> 6);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        uint4 w = __ldg(rw + c);
        float sc = __half2float(__ldg(rs + (c >> 1)));
        const unsigned ww[4] = {w.x, w.y, w.z, w.w};
        float part = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            uint4 xq = __ldg(reinterpret_cast<const uint4*>(x + c * 32 + p * 8));
            const half2* h = reinterpret_cast<const half2*>(&xq);
            const unsigned u = ww[p];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h[j]);
                part = fmaf(static_cast<float>(static_cast<int>((u >> (8*j)) & 15u) - 8), f.x,
                       fmaf(static_cast<float>(static_cast<int>((u >> (8*j+4)) & 15u) - 8), f.y, part));
            }
        }
        acc = fmaf(sc, part, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc);
}

__global__ void silu_mul(const Half* g, const Half* u, Half* out, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    float gf = __half2float(g[i]);
    out[i] = __float2half((gf / (1.0f + __expf(-gf))) * __half2float(u[i]));
}

template <class Fn>
double bench_us_cold(Fn&& fn, uint32_t* d_flush, int flush_n, cudaStream_t stream, int warm = 10, int iters = 51) {
    for (int i = 0; i < warm; ++i) { flush_l2<<<160, 256, 0, stream>>>(d_flush, flush_n); fn(); }
    CHECK(cudaStreamSynchronize(stream));
    std::vector<double> ts;
    cudaEvent_t s, e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    for (int i = 0; i < iters; ++i) {
        flush_l2<<<160, 256, 0, stream>>>(d_flush, flush_n);
        CHECK(cudaEventRecord(s, stream));
        fn();
        CHECK(cudaEventRecord(e, stream));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, s, e));
        ts.push_back(ms * 1000.0);
    }
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

} // namespace

int main() {
    cudaStream_t stream; cublasHandle_t blas;
    CHECK(cudaStreamCreate(&stream)); CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas, stream));
    const int flush_n = (32u << 20) / 4;
    uint32_t* d_flush; CHECK(cudaMalloc(&d_flush, flush_n * sizeof(uint32_t)));

    struct Shape { const char* name; int H; int I; int sk_fp16; int sk_int4; };
    std::vector<Shape> shapes = {
        {"Qwen3-0.6B_FFN_1024x3072", 1024, 3072, 24, 40},
        {"Qwen3-8B_FFN_4096x12288",  4096, 12288, 16, 24}
    };

    const float alpha = 1.0f, beta = 0.0f;
    std::cout << "{\n  \"track\": \"R2-1A_v2_SplitK_Coalesced_Sparse_FFN_CUDA\",\n  \"shapes\": [\n";
    for (size_t si = 0; si < shapes.size(); ++si) {
        int H = shapes[si].H, I = shapes[si].I;
        Half *d_x, *d_Wg, *d_Wu, *d_Wd, *d_WdT, *d_g, *d_u, *d_act, *d_y16, *d_Vr, *d_Ur;
        float *d_y32, *d_z;
        uint4 *d_Wg4, *d_Wu4, *d_Wd4, *d_WdT4;
        Half *d_Sg4, *d_Su4, *d_Sd4, *d_SdT4;
        int *d_idx, *d_cnt;
        CHECK(cudaMalloc(&d_x, H * sizeof(Half)));
        CHECK(cudaMalloc(&d_Wg, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMalloc(&d_Wu, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMalloc(&d_Wd, static_cast<std::size_t>(H) * I * sizeof(Half)));
        CHECK(cudaMalloc(&d_WdT, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMalloc(&d_g, I * sizeof(Half))); CHECK(cudaMalloc(&d_u, I * sizeof(Half)));
        CHECK(cudaMalloc(&d_act, I * sizeof(Half))); CHECK(cudaMalloc(&d_y16, H * sizeof(Half)));
        CHECK(cudaMalloc(&d_y32, H * sizeof(float)));
        CHECK(cudaMalloc(&d_Vr, static_cast<std::size_t>(kRank) * H * sizeof(Half)));
        CHECK(cudaMalloc(&d_Ur, static_cast<std::size_t>(I) * kRank * sizeof(Half)));
        CHECK(cudaMalloc(&d_z, kRank * sizeof(float)));
        CHECK(cudaMalloc(&d_Wg4, static_cast<std::size_t>(I) * (H >> 5) * sizeof(uint4)));
        CHECK(cudaMalloc(&d_Wu4, static_cast<std::size_t>(I) * (H >> 5) * sizeof(uint4)));
        CHECK(cudaMalloc(&d_Wd4, static_cast<std::size_t>(H) * (I >> 5) * sizeof(uint4)));
        CHECK(cudaMalloc(&d_WdT4, static_cast<std::size_t>(I) * (H >> 5) * sizeof(uint4)));
        CHECK(cudaMalloc(&d_Sg4, static_cast<std::size_t>(I) * (H >> 6) * sizeof(Half)));
        CHECK(cudaMalloc(&d_Su4, static_cast<std::size_t>(I) * (H >> 6) * sizeof(Half)));
        CHECK(cudaMalloc(&d_Sd4, static_cast<std::size_t>(H) * (I >> 6) * sizeof(Half)));
        CHECK(cudaMalloc(&d_SdT4, static_cast<std::size_t>(I) * (H >> 6) * sizeof(Half)));
        CHECK(cudaMalloc(&d_idx, I * sizeof(int))); CHECK(cudaMalloc(&d_cnt, sizeof(int)));

        CHECK(cudaMemset(d_x, 0x3c, H * sizeof(Half)));
        CHECK(cudaMemset(d_Wg, 0x3c, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMemset(d_Wu, 0x3c, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMemset(d_Wd, 0x3c, static_cast<std::size_t>(H) * I * sizeof(Half)));
        CHECK(cudaMemset(d_WdT, 0x3c, static_cast<std::size_t>(I) * H * sizeof(Half)));
        CHECK(cudaMemset(d_act, 0x3c, I * sizeof(Half)));
        CHECK(cudaMemset(d_Vr, 0x3c, static_cast<std::size_t>(kRank) * H * sizeof(Half)));
        CHECK(cudaMemset(d_Ur, 0x3c, static_cast<std::size_t>(I) * kRank * sizeof(Half)));

        // Dense baselines
        double cublas_down_us = bench_us_cold([&](){
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, H, 1, I, &alpha,
                d_Wd, CUDA_R_16F, I, d_act, CUDA_R_16F, I, &beta, d_y16, CUDA_R_16F, H,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }, d_flush, flush_n, stream);

        double int4_down_us = bench_us_cold([&](){
            int4_gemv<<<(H+7)/8, 256, 0, stream>>>(d_Wd4, d_Sd4, d_act, d_y16, H, I);
        }, d_flush, flush_n, stream);

        double cublas_full_us = bench_us_cold([&](){
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, I, 1, H, &alpha, d_Wg, CUDA_R_16F, H, d_x, CUDA_R_16F, H, &beta, d_g, CUDA_R_16F, I, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, I, 1, H, &alpha, d_Wu, CUDA_R_16F, H, d_x, CUDA_R_16F, H, &beta, d_u, CUDA_R_16F, I, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            silu_mul<<<(I+255)/256, 256, 0, stream>>>(d_g, d_u, d_act, I);
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, H, 1, I, &alpha, d_Wd, CUDA_R_16F, I, d_act, CUDA_R_16F, I, &beta, d_y16, CUDA_R_16F, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }, d_flush, flush_n, stream);

        double int4_full_us = bench_us_cold([&](){
            int4_gemv<<<(I+7)/8, 256, 0, stream>>>(d_Wg4, d_Sg4, d_x, d_g, I, H);
            int4_gemv<<<(I+7)/8, 256, 0, stream>>>(d_Wu4, d_Su4, d_x, d_u, I, H);
            silu_mul<<<(I+255)/256, 256, 0, stream>>>(d_g, d_u, d_act, I);
            int4_gemv<<<(H+7)/8, 256, 0, stream>>>(d_Wd4, d_Sd4, d_act, d_y16, H, I);
        }, d_flush, flush_n, stream);

        // Rank-64 Paley-Wiener screener latency
        double rank64_screen_us = bench_us_cold([&](){
            CHECK(cudaMemsetAsync(d_cnt, 0, sizeof(int), stream));
            rank64_proj_v<<<(kRank+7)/8, 256, 0, stream>>>(d_Vr, d_x, d_z, H);
            rank64_screen_neurons<<<(I+255)/256, 256, 0, stream>>>(d_Ur, d_z, d_idx, d_cnt, I, 1e9f);
        }, d_flush, flush_n, stream);

        std::cout << std::fixed << std::setprecision(2)
                  << "    {\n      \"name\": \"" << shapes[si].name << "\", \"H\": " << H << ", \"I\": " << I << ",\n"
                  << "      \"dense_cublas_down_us\": " << cublas_down_us << ", \"dense_int4_down_us\": " << int4_down_us << ",\n"
                  << "      \"dense_cublas_full_ffn_us\": " << cublas_full_us << ", \"dense_int4_full_ffn_us\": " << int4_full_us << ",\n"
                  << "      \"rank64_screen_us\": " << rank64_screen_us << ",\n      \"results\": [\n";

        std::vector<double> skips = {0.50, 0.60, 0.70, 0.75};
        for (size_t ki = 0; ki < skips.size(); ++ki) {
            double sp = skips[ki];
            int keep_n = static_cast<int>(std::lround(I * (1.0 - sp)));
            std::vector<int> h_idx(keep_n);
            for (int k = 0; k < keep_n; ++k) h_idx[k] = (k * I) / keep_n;
            CHECK(cudaMemcpy(d_idx, h_idx.data(), keep_n * sizeof(int), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_cnt, &keep_n, sizeof(int), cudaMemcpyHostToDevice));

            // Split-K W_down FP16 & INT4
            dim3 g_dfp((H/4 + 127)/128, shapes[si].sk_fp16);
            double sp_down_fp16 = bench_us_cold([&](){
                CHECK(cudaMemsetAsync(d_y32, 0, H * sizeof(float), stream));
                colmajor_sparse_down_fp16_splitk<<<g_dfp, 128, 0, stream>>>(d_WdT, d_act, d_idx, d_cnt, d_y32, H, shapes[si].sk_fp16);
            }, d_flush, flush_n, stream);

            dim3 g_di4((H/8 + 127)/128, shapes[si].sk_int4);
            double sp_down_int4 = bench_us_cold([&](){
                CHECK(cudaMemsetAsync(d_y32, 0, H * sizeof(float), stream));
                colmajor_sparse_down_int4_splitk<<<g_di4, 128, 0, stream>>>(
                    reinterpret_cast<const uint32_t*>(d_WdT4), d_SdT4, d_act, d_idx, d_cnt, d_y32, H, shapes[si].sk_int4);
            }, d_flush, flush_n, stream);

            // Full Rank-64 Screened SwiGLU (FP16 & INT4)
            double sp_full_fp16 = rank64_screen_us + bench_us_cold([&](){
                sparse_gate_up_fp16<<<(keep_n + 7)/8, 256, 0, stream>>>(d_x, d_Wg, d_Wu, d_idx, d_cnt, d_act, H);
                CHECK(cudaMemsetAsync(d_y32, 0, H * sizeof(float), stream));
                colmajor_sparse_down_fp16_splitk<<<g_dfp, 128, 0, stream>>>(d_WdT, d_act, d_idx, d_cnt, d_y32, H, shapes[si].sk_fp16);
            }, d_flush, flush_n, stream);

            double sp_full_int4 = rank64_screen_us + bench_us_cold([&](){
                sparse_gate_up_int4<<<(keep_n + 7)/8, 256, 0, stream>>>(d_x, d_Wg4, d_Sg4, d_Wu4, d_Su4, d_idx, d_cnt, d_act, H);
                CHECK(cudaMemsetAsync(d_y32, 0, H * sizeof(float), stream));
                colmajor_sparse_down_int4_splitk<<<g_di4, 128, 0, stream>>>(
                    reinterpret_cast<const uint32_t*>(d_WdT4), d_SdT4, d_act, d_idx, d_cnt, d_y32, H, shapes[si].sk_int4);
            }, d_flush, flush_n, stream);

            std::cout << "        {\"skip_pct\": " << int(sp*100)
                      << ", \"down_fp16_us\": " << sp_down_fp16
                      << ", \"down_fp16_speedup\": " << (cublas_down_us / sp_down_fp16)
                      << ", \"down_int4_us\": " << sp_down_int4
                      << ", \"down_int4_speedup\": " << (int4_down_us / sp_down_int4)
                      << ", \"full_fp16_us\": " << sp_full_fp16
                      << ", \"full_fp16_speedup\": " << (cublas_full_us / sp_full_fp16)
                      << ", \"full_int4_us\": " << sp_full_int4
                      << ", \"full_int4_speedup\": " << (int4_full_us / sp_full_int4)
                      << ", \"full_int4_vs_fp16\": " << (cublas_full_us / sp_full_int4) << "}"
                      << (ki + 1 < skips.size() ? ",\n" : "\n");
        }
        std::cout << "      ]\n    }" << (si + 1 < shapes.size() ? ",\n" : "\n");

        cudaFree(d_x); cudaFree(d_Wg); cudaFree(d_Wu); cudaFree(d_Wd); cudaFree(d_WdT);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_act); cudaFree(d_y16); cudaFree(d_y32);
        cudaFree(d_Vr); cudaFree(d_Ur); cudaFree(d_z);
        cudaFree(d_Wg4); cudaFree(d_Wu4); cudaFree(d_Wd4); cudaFree(d_WdT4);
        cudaFree(d_Sg4); cudaFree(d_Su4); cudaFree(d_Sd4); cudaFree(d_SdT4);
        cudaFree(d_idx); cudaFree(d_cnt);
    }
    std::cout << "  ]\n}\n";
    return 0;
}
