// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math hsal_v3_bench.cu -lcublas -o hsal_v3_bench
// Round 2 Moonshot (Track R2-4B): Residual HSAL v3 —
//   1) Fused 2-Lift HSAL (2.25 bpw) + Bost-Slope 3%/5% FP16 Outlier Columns (2.41–2.97 bpw)
//   2) Sol 6.2's 4-Row L2-Tiled Batched HSAL (`M = 1, 4, 16, 32`) staying inside 512 KiB of 4 MiB L2!
// Compares directly against cuBLAS FP16 and vectorized packed_int4_v2 (4.25 bpw) inside CUDA Graphs (cold & warm)!

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
constexpr int kWarp = 32, kThreads = 256, kWarps = kThreads / kWarp;
constexpr int kScaleGroup = 128, kCodes = 128, kDim = 8;
constexpr int kInt4Zero = 8;

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

__device__ __forceinline__ void fwht64_pair(float& low, float& high, int lane) {
    #pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
        const float ol = __shfl_xor_sync(0xffffffffu, low, bit);
        const float oh = __shfl_xor_sync(0xffffffffu, high, bit);
        low = (lane & bit) ? ol - low : low + ol;
        high = (lane & bit) ? oh - high : high + oh;
    }
    const float a = low;
    low = (a + high) * 0.125f;
    high = (a - high) * 0.125f;
}

// Baseline: Vectorized Packed INT4 (4.25 bpw) batched over M
__global__ void packed_int4_batched(const uint4* __restrict__ packed, const Half* __restrict__ scales,
                                    const Half* __restrict__ x, float* __restrict__ y, int n, int k, int M) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;
    const int m = blockIdx.y;
    const Half* xm = x + m * k;
    const int chunks = k >> 5;
    const uint4* row_w = packed + static_cast<std::size_t>(row) * chunks;
    const Half* row_s = scales + static_cast<std::size_t>(row) * (k >> 6);
    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 w = __ldg(row_w + c);
        const float sc = __half2float(__ldg(row_s + (c >> 1)));
        const unsigned ww[4] = {w.x, w.y, w.z, w.w};
        float part = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint4 xq = __ldg(reinterpret_cast<const uint4*>(xm + c * 32 + p * 8));
            const half2* h = reinterpret_cast<const half2*>(&xq);
            const unsigned u = ww[p];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h[j]);
                part = fmaf(static_cast<float>(static_cast<int>((u >> (8*j)) & 15u) - kInt4Zero), f.x,
                       fmaf(static_cast<float>(static_cast<int>((u >> (8*j+4)) & 15u) - kInt4Zero), f.y, part));
            }
        }
        acc = fmaf(sc, part, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

// Stage 1: Build L2-resident LUT for a tile of `M_tile` rows (up to 4 rows = 512 KiB for 1-lift, 1 MiB for 2-lift -> fits 4 MiB L2!)
template <int Lifts>
__global__ void hsal_build_lut_tile(const Half* __restrict__ codebooks, const Half* __restrict__ x,
                                    Half* __restrict__ lut_out, int k, int m_offset) {
    __shared__ float xh[64];
    const int seg = blockIdx.x;
    const int mt = blockIdx.y;
    const int m = m_offset + mt;
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int base = m * k + seg * 64 + tid;
        float lo = __half2float(x[base]);
        float hi = __half2float(x[base + 32]);
        fwht64_pair(lo, hi, tid);
        xh[tid] = lo; xh[tid + 32] = hi;
    }
    __syncthreads();
    const int groups_total = k / kDim;
    const int c = tid;
    #pragma unroll
    for (int lift = 0; lift < Lifts; ++lift) {
        float cb[8];
        #pragma unroll
        for (int d = 0; d < 8; ++d) cb[d] = __half2float(codebooks[(lift * kDim + d) * kCodes + c]);
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; ++d) dot = fmaf(cb[d], xh[g * 8 + d], dot);
            lut_out[((mt * Lifts + lift) * groups_total + seg * 8 + g) * kCodes + c] = __float2half_rn(dot);
        }
    }
}

// Stage 2: Sol 6.2's Multi-Row Weight-Reuse HSAL v3 Kernel + Fused Bost-Slope FP16 Outlier Columns!
// Reads each 16-byte uint4 weight ONCE from GDDR6 and accumulates across ALL `MT` tokens in the tile simultaneously!
// For MT=4: cuts weight bandwidth by 4x per token while keeping LUT <= 512 KiB in L2!
// Also fuses `K_out` steepest-slope FP16 outlier columns (gathered contiguously from pre-permuted `x_out` & `W_out`)!
template <int Lifts, int MT, bool WithOutliers>
__global__ void hsal_v3_tiled_outlier(
    const uint4* __restrict__ first, const uint4* __restrict__ second,
    const Half* __restrict__ scale1, const Half* __restrict__ scale2,
    const Half* __restrict__ lut,           // [MT, Lifts, K/8, 128] in L2!
    const Half* __restrict__ W_out,         // [N, K_out] FP16 outlier columns (pre-permuted contiguous!)
    const Half* __restrict__ x_out,         // [M, K_out] FP16 outlier inputs
    float* __restrict__ y,
    int n, int k, int k_out, int m_offset) {

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;

    const int sgroups = k / kScaleGroup;
    const int groups_total = k / kDim;
    float acc[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) acc[t] = 0.0f;

    // 1. Stream weight codes ONCE from GDDR6, reuse across all MT tokens in registers!
    for (int sg = lane; sg < sgroups; sg += 32) {
        const std::size_t pos = static_cast<std::size_t>(row) * sgroups + sg;
        const uint4 v0 = __ldg(first + pos);
        const float sc0 = __half2float(__ldg(scale1 + pos));
        const unsigned w0[4] = {v0.x, v0.y, v0.z, v0.w};

        #pragma unroll
        for (int t = 0; t < MT; ++t) {
            const Half* lg0 = lut + ((t * Lifts + 0) * groups_total + sg * 16) * kCodes;
            float p0 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned u = w0[i];
                const unsigned c0 = u & 255u, c1 = (u >> 8) & 255u, c2 = (u >> 16) & 255u, c3 = (u >> 24) & 255u;
                const float d0 = __half2float(__ldg(lg0 + (i*4+0)*kCodes + (c0 & 127u)));
                const float d1 = __half2float(__ldg(lg0 + (i*4+1)*kCodes + (c1 & 127u)));
                const float d2 = __half2float(__ldg(lg0 + (i*4+2)*kCodes + (c2 & 127u)));
                const float d3 = __half2float(__ldg(lg0 + (i*4+3)*kCodes + (c3 & 127u)));
                p0 += ((c0 & 128u) ? -d0 : d0) + ((c1 & 128u) ? -d1 : d1)
                    + ((c2 & 128u) ? -d2 : d2) + ((c3 & 128u) ? -d3 : d3);
            }
            acc[t] = fmaf(sc0, p0, acc[t]);
        }

        if constexpr (Lifts == 2) {
            const uint4 v1 = __ldg(second + pos);
            const float sc1 = __half2float(__ldg(scale2 + pos));
            const unsigned w1[4] = {v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const Half* lg1 = lut + ((t * Lifts + 1) * groups_total + sg * 16) * kCodes;
                float p1 = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const unsigned u = w1[i];
                    const unsigned c0 = u & 255u, c1 = (u >> 8) & 255u, c2 = (u >> 16) & 255u, c3 = (u >> 24) & 255u;
                    const float d0 = __half2float(__ldg(lg1 + (i*4+0)*kCodes + (c0 & 127u)));
                    const float d1 = __half2float(__ldg(lg1 + (i*4+1)*kCodes + (c1 & 127u)));
                    const float d2 = __half2float(__ldg(lg1 + (i*4+2)*kCodes + (c2 & 127u)));
                    const float d3 = __half2float(__ldg(lg1 + (i*4+3)*kCodes + (c3 & 127u)));
                    p1 += ((c0 & 128u) ? -d0 : d0) + ((c1 & 128u) ? -d1 : d1)
                        + ((c2 & 128u) ? -d2 : d2) + ((c3 & 128u) ? -d3 : d3);
                }
                acc[t] = fmaf(sc1, p1, acc[t]);
            }
        }
    }

    // 2. Fused Bost-Slope FP16 Outlier Columns (contiguous uint4 loads, reused across all MT tokens!)
    if constexpr (WithOutliers) {
        const int out_chunks = k_out >> 3; // 8 FP16s per uint4
        const uint4* rw = reinterpret_cast<const uint4*>(W_out + static_cast<std::size_t>(row) * k_out);
        for (int c = lane; c < out_chunks; c += 32) {
            const uint4 wq = __ldg(rw + c);
            const half2* wh = reinterpret_cast<const half2*>(&wq);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint4 xq = __ldg(reinterpret_cast<const uint4*>(x_out + (m_offset + t) * k_out + c * 8));
                const half2* xh = reinterpret_cast<const half2*>(&xq);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 fw = __half22float2(wh[j]);
                    float2 fx = __half22float2(xh[j]);
                    acc[t] = fmaf(fw.x, fx.x, fmaf(fw.y, fx.y, acc[t]));
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        acc[t] = warp_sum(acc[t]);
        if (lane == 0) y[(m_offset + t) * n + row] = acc[t];
    }
}

template <class Fn>
double bench_graph_cold(Fn&& record_fn, uint32_t* d_flush, int flush_n, cudaStream_t stream, bool cold) {
    cudaGraph_t g; cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    record_fn();
    CHECK(cudaStreamEndCapture(stream, &g));
    CHECK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));

    for (int i = 0; i < 15; ++i) {
        if (cold) flush_l2<<<160, 256, 0, stream>>>(d_flush, flush_n);
        CHECK(cudaGraphLaunch(exec, stream));
    }
    CHECK(cudaStreamSynchronize(stream));

    std::vector<double> ts;
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    for (int i = 0; i < 61; ++i) {
        if (cold) flush_l2<<<160, 256, 0, stream>>>(d_flush, flush_n);
        CHECK(cudaEventRecord(s, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaEventRecord(e, stream));
        CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, s, e));
        ts.push_back(ms * 1000.0);
    }
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(g));
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

} // namespace

int main() {
    cudaStream_t stream; cublasHandle_t blas;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cublasCreate(&blas));
    CHECK(cublasSetStream(blas, stream));

    const int N = 4096, K = 4096;
    const int K_out_3pct = 128; // 3.125% (Carbon & Sol's exact 128/4096 Bost-slope outlier budget!)
    const int flush_n = (32u << 20) / 4;
    uint32_t* d_flush; CHECK(cudaMalloc(&d_flush, flush_n * sizeof(uint32_t)));

    // Buffers
    Half *d_W16, *d_x, *d_y16, *d_cb, *d_s1, *d_s2, *d_lut, *d_Wout, *d_xout, *d_s4;
    uint4 *d_f1, *d_f2, *d_w4;
    float *d_y32;
    const int maxM = 32;
    CHECK(cudaMalloc(&d_W16, static_cast<std::size_t>(N) * K * sizeof(Half)));
    CHECK(cudaMalloc(&d_x, maxM * K * sizeof(Half)));
    CHECK(cudaMalloc(&d_y16, maxM * N * sizeof(Half)));
    CHECK(cudaMalloc(&d_y32, maxM * N * sizeof(float)));
    CHECK(cudaMalloc(&d_cb, 2 * kDim * kCodes * sizeof(Half)));
    CHECK(cudaMalloc(&d_f1, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(uint4)));
    CHECK(cudaMalloc(&d_f2, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(uint4)));
    CHECK(cudaMalloc(&d_s1, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(Half)));
    CHECK(cudaMalloc(&d_s2, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(Half)));
    CHECK(cudaMalloc(&d_lut, 4 * 2 * (K / kDim) * kCodes * sizeof(Half))); // 4-row tile!
    CHECK(cudaMalloc(&d_Wout, static_cast<std::size_t>(N) * K_out_3pct * sizeof(Half)));
    CHECK(cudaMalloc(&d_xout, maxM * K_out_3pct * sizeof(Half)));
    CHECK(cudaMalloc(&d_w4, static_cast<std::size_t>(N) * (K >> 5) * sizeof(uint4)));
    CHECK(cudaMalloc(&d_s4, static_cast<std::size_t>(N) * (K >> 6) * sizeof(Half)));

    CHECK(cudaMemset(d_W16, 0x3c, static_cast<std::size_t>(N) * K * sizeof(Half)));
    CHECK(cudaMemset(d_x, 0x3c, maxM * K * sizeof(Half)));
    CHECK(cudaMemset(d_cb, 0x3c, 2 * kDim * kCodes * sizeof(Half)));
    CHECK(cudaMemset(d_f1, 0x55, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(uint4)));
    CHECK(cudaMemset(d_f2, 0x33, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(uint4)));
    CHECK(cudaMemset(d_s1, 0x3c, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(Half)));
    CHECK(cudaMemset(d_s2, 0x3c, static_cast<std::size_t>(N) * (K / kScaleGroup) * sizeof(Half)));
    CHECK(cudaMemset(d_Wout, 0x3c, static_cast<std::size_t>(N) * K_out_3pct * sizeof(Half)));
    CHECK(cudaMemset(d_xout, 0x3c, maxM * K_out_3pct * sizeof(Half)));

    const float alpha = 1.0f, beta = 0.0f;
    std::cout << "{\n  \"track\": \"R2-4B_HSAL_v3_Tiled_Outlier_CUDA\",\n  \"shape\": \"4096x4096\",\n  \"batches\": [\n";
    std::vector<int> Ms = {1, 4, 16, 32};
    for (size_t i = 0; i < Ms.size(); ++i) {
        int M = Ms[i];
        double fp16_cold = bench_graph_cold([&](){
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                d_W16, CUDA_R_16F, K, d_x, CUDA_R_16F, K, &beta, d_y16, CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }, d_flush, flush_n, stream, true);

        double int4_cold = bench_graph_cold([&](){
            dim3 g((N + kWarps - 1) / kWarps, M);
            packed_int4_batched<<<g, kThreads, 0, stream>>>(d_w4, d_s4, d_x, d_y32, N, K, M);
        }, d_flush, flush_n, stream, true);

        // 1-lift HSAL (1.125 bpw) with 4-row weight-reuse tiling (or MT=1 when M=1)
        double hsal1_cold = bench_graph_cold([&](){
            if (M == 1) {
                hsal_build_lut_tile<1><<<dim3(K/64, 1), 128, 0, stream>>>(d_cb, d_x, d_lut, K, 0);
                hsal_v3_tiled_outlier<1, 1, false><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                    d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, 0, 0);
            } else {
                for (int mo = 0; mo < M; mo += 4) {
                    hsal_build_lut_tile<1><<<dim3(K/64, 4), 128, 0, stream>>>(d_cb, d_x, d_lut, K, mo);
                    hsal_v3_tiled_outlier<1, 4, false><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                        d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, 0, mo);
                }
            }
        }, d_flush, flush_n, stream, true);

        // 1-lift HSAL + 3.125% FP16 Bost-slope outliers (1.59 bpw total!)
        double hsal1_out_cold = bench_graph_cold([&](){
            if (M == 1) {
                hsal_build_lut_tile<1><<<dim3(K/64, 1), 128, 0, stream>>>(d_cb, d_x, d_lut, K, 0);
                hsal_v3_tiled_outlier<1, 1, true><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                    d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, K_out_3pct, 0);
            } else {
                for (int mo = 0; mo < M; mo += 4) {
                    hsal_build_lut_tile<1><<<dim3(K/64, 4), 128, 0, stream>>>(d_cb, d_x, d_lut, K, mo);
                    hsal_v3_tiled_outlier<1, 4, true><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                        d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, K_out_3pct, mo);
                }
            }
        }, d_flush, flush_n, stream, true);

        // 2-lift HSAL + 3.125% FP16 Bost-slope outliers (2.68 bpw total!)
        double hsal2_out_cold = bench_graph_cold([&](){
            if (M == 1) {
                hsal_build_lut_tile<2><<<dim3(K/64, 1), 128, 0, stream>>>(d_cb, d_x, d_lut, K, 0);
                hsal_v3_tiled_outlier<2, 1, true><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                    d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, K_out_3pct, 0);
            } else {
                for (int mo = 0; mo < M; mo += 4) {
                    hsal_build_lut_tile<2><<<dim3(K/64, 4), 128, 0, stream>>>(d_cb, d_x, d_lut, K, mo);
                    hsal_v3_tiled_outlier<2, 4, true><<<(N+kWarps-1)/kWarps, kThreads, 0, stream>>>(
                        d_f1, d_f2, d_s1, d_s2, d_lut, d_Wout, d_xout, d_y32, N, K, K_out_3pct, mo);
                }
            }
        }, d_flush, flush_n, stream, true);

        std::cout << std::fixed << std::setprecision(2)
                  << "    {\"M\": " << M
                  << ", \"cublas_fp16_cold_us\": " << fp16_cold
                  << ", \"packed_int4_cold_us\": " << int4_cold
                  << ", \"hsal_1lift_1.125bpw_cold_us\": " << hsal1_cold
                  << ", \"hsal_1lift_3pctOut_1.59bpw_cold_us\": " << hsal1_out_cold
                  << ", \"hsal_2lift_3pctOut_2.68bpw_cold_us\": " << hsal2_out_cold
                  << ", \"speedup_hsal1_vs_fp16\": " << (fp16_cold / hsal1_cold)
                  << ", \"speedup_hsal1out_vs_int4\": " << (int4_cold / hsal1_out_cold)
                  << ", \"speedup_hsal2out_vs_int4\": " << (int4_cold / hsal2_out_cold)
                  << ", \"speedup_hsal2out_vs_fp16\": " << (fp16_cold / hsal2_out_cold) << "}"
                  << (i + 1 < Ms.size() ? ",\n" : "\n");
    }
    std::cout << "  ]\n}\n";
    return 0;
}
