// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math reg_sign_bench.cu -lcublas -o reg_sign_bench
// Tests whether ZERO-LUT direct register sign-bit unpacking (1.125 bpw sign-only & 2.5 bpw dual-sign)
// smashes the random-LUT L1 gather bottleneck discovered in hsal_bench_v2!
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
using Byte = std::uint8_t;
constexpr int kThreads = 256, kWarps = 8, kSamples = 101, kWarmups = 15;
constexpr std::size_t kFlushBytes = 32u << 20;

void check(cudaError_t s, const char* c) { if (s != cudaSuccess) throw std::runtime_error(std::string(c) + ": " + cudaGetErrorString(s)); }
void check(cublasStatus_t s, const char* c) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(c) + ": " + cublasGetStatusString(s)); }
#define CHECK(x) check((x), #x)

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}
__global__ void flush_l2(std::uint32_t* buf, int words) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += blockDim.x * gridDim.x) buf[i] += 0x9e3779b9u;
}

// 1.125 bpw Sign-Only Direct Register Kernel (0 LUT gathers! Coalesced uint4 = 128 sign bits + 1 FP16 scale!)
// Wait: 128 weights = 16 bytes = 1 uint4! Across 32 lanes = 4096 weights in ONE SINGLE LOAD!
__global__ void reg_sign1_gemv(const uint4* __restrict__ signs, const Half* __restrict__ scales,
                               const Half* __restrict__ x, float* __restrict__ y, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;
    const int m = blockIdx.y;
    const Half* xm = x + m * k;
    const int groups = k >> 7;  // k / 128

    float acc = 0.0f;
    for (int g = lane; g < groups; g += 32) {
        const std::size_t pos = static_cast<std::size_t>(row) * groups + g;
        const uint4 bits = __ldg(signs + pos);
        const float sc = __half2float(__ldg(scales + pos));
        const unsigned w[4] = {bits.x, bits.y, bits.z, bits.w};
        float part = 0.0f;
        const uint4* xp = reinterpret_cast<const uint4*>(xm + g * 128);
        #pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            unsigned mask = w[wi];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 xv = __ldg(xp + wi * 4 + c);
                const half2* h = reinterpret_cast<const half2*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h[j]);
                    const unsigned b0 = (mask & 1u);
                    const unsigned b1 = (mask & 2u);
                    mask >>= 2;
                    part += (b0 ? -f.x : f.x) + (b1 ? -f.y : f.y);
                }
            }
        }
        acc = fmaf(sc, part, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

// 2.5 bpw Dual-Sign Direct Register Kernel (ZERO LUT gathers! 2 sign planes + half2 scales per 64 weights!)
// 64 weights = 8 bytes plane1 + 8 bytes plane2 = 16 bytes (1 uint4: .x/.y = plane1 64 bits, .z/.w = plane2 64 bits!)
__global__ void reg_dual_sign_gemv(const uint4* __restrict__ planes, const half2* __restrict__ scales,
                                   const Half* __restrict__ x, float* __restrict__ y, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;
    const int m = blockIdx.y;
    const Half* xm = x + m * k;
    const int groups = k >> 6;  // k / 64

    float acc = 0.0f;
    for (int g = lane; g < groups; g += 32) {
        const std::size_t pos = static_cast<std::size_t>(row) * groups + g;
        const uint4 p = __ldg(planes + pos);
        const float2 sc = __half22float2(__ldg(scales + pos));
        const unsigned p1[2] = {p.x, p.y};
        const unsigned p2[2] = {p.z, p.w};
        float s1 = 0.0f, s2 = 0.0f;
        const uint4* xp = reinterpret_cast<const uint4*>(xm + g * 64);
        #pragma unroll
        for (int wi = 0; wi < 2; ++wi) {
            unsigned m1 = p1[wi], m2 = p2[wi];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 xv = __ldg(xp + wi * 4 + c);
                const half2* h = reinterpret_cast<const half2*>(&xv);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h[j]);
                    s1 += ((m1 & 1u) ? -f.x : f.x) + ((m1 & 2u) ? -f.y : f.y);
                    s2 += ((m2 & 1u) ? -f.x : f.x) + ((m2 & 2u) ? -f.y : f.y);
                    m1 >>= 2; m2 >>= 2;
                }
            }
        }
        acc = fmaf(sc.x, s1, fmaf(sc.y, s2, acc));
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

template <class Fn>
float time_graph(cudaStream_t st, Fn&& launch, std::uint32_t* flush, int sms, bool cold) {
    cudaGraph_t g = nullptr; cudaGraphExec_t ex = nullptr;
    CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    launch();
    CHECK(cudaStreamEndCapture(st, &g));
    CHECK(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0));
    cudaEvent_t s, e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    for (int i = 0; i < kWarmups; ++i) {
        if (cold) flush_l2<<<sms * 4, 256, 0, st>>>(flush, kFlushBytes / 4);
        CHECK(cudaGraphLaunch(ex, st));
    }
    CHECK(cudaStreamSynchronize(st));
    std::vector<float> ts(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        if (cold) flush_l2<<<sms * 4, 256, 0, st>>>(flush, kFlushBytes / 4);
        CHECK(cudaEventRecord(s, st));
        CHECK(cudaGraphLaunch(ex, st));
        CHECK(cudaEventRecord(e, st));
        CHECK(cudaEventSynchronize(e));
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, s, e));
        ts[i] = ms * 1000.0f;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
    std::sort(ts.begin(), ts.end());
    return ts[kSamples / 2];
}
}  // namespace

int main() {
    cudaDeviceProp dev{}; CHECK(cudaGetDeviceProperties(&dev, 0));
    cudaStream_t st; CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cublasHandle_t blas; CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas, st));
    void* ws = nullptr; CHECK(cudaMalloc(&ws, 8u << 20)); CHECK(cublasSetWorkspace(blas, ws, 8u << 20));
    std::uint32_t* flush = nullptr; CHECK(cudaMalloc(&flush, kFlushBytes));
    CHECK(cudaMemsetAsync(flush, 0, kFlushBytes, st)); CHECK(cudaStreamSynchronize(st));

    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0f, 0.025f);
    std::cout << "{\n  \"results\":[\n";
    bool first = true;
    for (auto [n, k] : {std::pair{4096, 4096}, std::pair{3072, 1024}}) {
        std::vector<Half> hW(n * k), hX(k), hS1(n * k / 128);
        std::vector<uint4> hSign1(n * k / 128, {0,0,0,0}), hDual(n * k / 64, {0,0,0,0});
        std::vector<half2> hSc2(n * k / 64);
        for (auto& v : hW) v = __float2half_rn(norm(rng));
        for (auto& v : hX) v = __float2half_rn(norm(rng) * 40.0f);
        // Pack sign1 (group 128)
        for (int r = 0; r < n; ++r) {
            for (int g = 0; g < k / 128; ++g) {
                float ma = 0;
                for (int i = 0; i < 128; ++i) ma += std::abs(__half2float(hW[r * k + g * 128 + i]));
                hS1[r * (k / 128) + g] = __float2half_rn(ma / 128.0f);
                unsigned w[4] = {0,0,0,0};
                for (int i = 0; i < 128; ++i)
                    if (__half2float(hW[r * k + g * 128 + i]) < 0.0f) w[i / 32] |= (1u << (i % 32));
                hSign1[r * (k / 128) + g] = {w[0], w[1], w[2], w[3]};
            }
            for (int g = 0; g < k / 64; ++g) {
                float ma = 0;
                for (int i = 0; i < 64; ++i) ma += std::abs(__half2float(hW[r * k + g * 64 + i]));
                float a1 = __half2float(__float2half_rn(ma / 64.0f));
                float rma = 0; unsigned p1[2] = {0,0}, p2[2] = {0,0};
                for (int i = 0; i < 64; ++i) {
                    float val = __half2float(hW[r * k + g * 64 + i]);
                    if (val < 0.0f) p1[i / 32] |= (1u << (i % 32));
                    float res = val - (val < 0.0f ? -a1 : a1);
                    if (res < 0.0f) p2[i / 32] |= (1u << (i % 32));
                    rma += std::abs(res);
                }
                hDual[r * (k / 64) + g] = {p1[0], p1[1], p2[0], p2[1]};
                hSc2[r * (k / 64) + g] = __halves2half2(__float2half_rn(a1), __float2half_rn(rma / 64.0f));
            }
        }
        Half *dW, *dX, *dS1; uint4 *dSign1, *dDual; half2 *dSc2; float *dY;
        CHECK(cudaMalloc(&dW, hW.size()*2)); CHECK(cudaMemcpy(dW, hW.data(), hW.size()*2, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dX, hX.size()*2)); CHECK(cudaMemcpy(dX, hX.data(), hX.size()*2, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dS1, hS1.size()*2)); CHECK(cudaMemcpy(dS1, hS1.data(), hS1.size()*2, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dSign1, hSign1.size()*16)); CHECK(cudaMemcpy(dSign1, hSign1.data(), hSign1.size()*16, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dDual, hDual.size()*16)); CHECK(cudaMemcpy(dDual, hDual.data(), hDual.size()*16, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dSc2, hSc2.size()*4)); CHECK(cudaMemcpy(dSc2, hSc2.data(), hSc2.size()*4, cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&dY, n * 4));

        float base_c = time_graph(st, [&]{
            float a=1.f, b=0.f;
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &a, dW, CUDA_R_16F, k, dX, CUDA_R_16F, k, &b, dY, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }, flush, dev.multiProcessorCount, true);
        float base_w = time_graph(st, [&]{
            float a=1.f, b=0.f;
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &a, dW, CUDA_R_16F, k, dX, CUDA_R_16F, k, &b, dY, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }, flush, dev.multiProcessorCount, false);

        auto run_one = [&](const char* name, std::size_t bytes, auto fn, auto verify_fn) {
            float c = time_graph(st, fn, flush, dev.multiProcessorCount, true);
            float w = time_graph(st, fn, flush, dev.multiProcessorCount, false);
            fn(); CHECK(cudaStreamSynchronize(st));
            std::vector<float> got(n); CHECK(cudaMemcpy(got.data(), dY, n*4, cudaMemcpyDeviceToHost));
            float cos = verify_fn(got);
            if (!first) std::cout << ",\n";
            first = false;
            std::cout << "    {\"method\":\"" << name << "\",\"n\":" << n << ",\"k\":" << k
                      << ",\"bytes\":" << bytes << ",\"bpw\":" << (8.0*bytes/(n*k))
                      << ",\"cold_us\":" << std::fixed << std::setprecision(2) << c
                      << ",\"cold_gbps\":" << std::setprecision(1) << (bytes/(1000.0*c))
                      << ",\"cold_speedup\":" << std::setprecision(2) << (base_c/c)
                      << ",\"warm_us\":" << std::setprecision(2) << w
                      << ",\"warm_gbps\":" << std::setprecision(1) << (bytes/(1000.0*w))
                      << ",\"warm_speedup\":" << std::setprecision(2) << (base_w/w)
                      << ",\"cos\":" << std::setprecision(7) << cos << "}";
        };
        run_one("cublas_fp16", hW.size()*2, [&]{
            float a=1.f, b=0.f;
            CHECK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &a, dW, CUDA_R_16F, k, dX, CUDA_R_16F, k, &b, dY, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }, [&](const std::vector<float>&){ return 1.0f; });

        run_one("reg_sign1_1.125bpw", hSign1.size()*16 + hS1.size()*2, [&]{
            reg_sign1_gemv<<<dim3((n + kWarps - 1)/kWarps, 1), kThreads, 0, st>>>(dSign1, dS1, dX, dY, n, k);
        }, [&](const std::vector<float>& got){
            double dot=0, nr=0, ng=0;
            for (int r=0; r<n; ++r) {
                double ref=0;
                for (int g=0; g<k/128; ++g) {
                    float sc = __half2float(hS1[r*(k/128)+g]);
                    uint4 u = hSign1[r*(k/128)+g]; unsigned ww[4]={u.x,u.y,u.z,u.w};
                    double p=0;
                    for (int i=0; i<128; ++i) { float x=__half2float(hX[g*128+i]); p += ((ww[i/32]>>(i%32))&1) ? -x : x; }
                    ref += sc * p;
                }
                dot += ref * got[r]; nr += ref*ref; ng += (double)got[r]*got[r];
            }
            return (float)(dot / (std::sqrt(nr)*std::sqrt(ng)));
        });

        run_one("reg_dual_sign_2.5bpw", hDual.size()*16 + hSc2.size()*4, [&]{
            reg_dual_sign_gemv<<<dim3((n + kWarps - 1)/kWarps, 1), kThreads, 0, st>>>(dDual, dSc2, dX, dY, n, k);
        }, [&](const std::vector<float>& got){
            double dot=0, nr=0, ng=0;
            for (int r=0; r<n; ++r) {
                double ref=0;
                for (int g=0; g<k/64; ++g) {
                    float2 sc = __half22float2(hSc2[r*(k/64)+g]);
                    uint4 u = hDual[r*(k/64)+g]; unsigned p1[2]={u.x,u.y}, p2[2]={u.z,u.w};
                    double s1=0, s2=0;
                    for (int i=0; i<64; ++i) {
                        float x=__half2float(hX[g*64+i]);
                        s1 += ((p1[i/32]>>(i%32))&1) ? -x : x;
                        s2 += ((p2[i/32]>>(i%32))&1) ? -x : x;
                    }
                    ref += sc.x*s1 + sc.y*s2;
                }
                dot += ref * got[r]; nr += ref*ref; ng += (double)got[r]*got[r];
            }
            return (float)(dot / (std::sqrt(nr)*std::sqrt(ng)));
        });

        cudaFree(dW); cudaFree(dX); cudaFree(dS1); cudaFree(dSign1); cudaFree(dDual); cudaFree(dSc2); cudaFree(dY);
    }
    std::cout << "\n  ]\n}\n";
    return 0;
}
