// nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math hsal_bench_v2.cu -lcublas -o hsal_bench_v2
// True Silicon-Optimal Rallis-Seesaw Benchmark on RTX 3070 Laptop (sm_86, 40 SMs, 4 MiB L2, 100 KiB SMEM/SM)
// Fixes v1's 128x-512x redundant LUT builds by separating 1-shot L2-resident LUT construction
// (and persistent 40-SM SMEM-LUT) from high-occupancy vectorized uint4 streaming!
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Half = __half;
using Byte = std::uint8_t;
constexpr int kWarp = 32, kThreads = 256, kWarps = kThreads / kWarp;
constexpr int kScaleGroup = 128, kCodes = 128, kDim = 8;
constexpr int kSamples = 101, kWarmups = 15;
constexpr std::size_t kFlushBytes = 32u << 20;
constexpr int kInt4Zero = 8;

void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
void check(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + cublasGetStatusString(status));
}
#define CHECK(call) check((call), #call)

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        CHECK(cudaMemcpy(data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

struct Context {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    DeviceBuffer<Byte> workspace{8u << 20};
    Context() {
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cublasSetWorkspace(blas, workspace.get(), 8u << 20));
    }
    ~Context() {
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
};

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

__global__ void flush_l2(std::uint32_t* buffer, int words) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += blockDim.x * gridDim.x)
        buffer[i] += 0x9e3779b9u;
}

// Fast vectorized Packed INT4 (4.25 bpw, group=64) with uint4 weights + uint4 x loads
__global__ void packed_int4_v2(const uint4* __restrict__ packed, const Half* __restrict__ scales,
                               const Half* __restrict__ x, float* __restrict__ y, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;

    const int m = blockIdx.y;
    const Half* xm = x + m * k;
    const int chunks = k >> 5;  // k / 32 (each uint4 = 16 bytes = 32 int4 weights)
    const uint4* row_w = packed + static_cast<std::size_t>(row) * chunks;
    const Half* row_s = scales + static_cast<std::size_t>(row) * (k >> 6);

    float acc = 0.0f;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 w = __ldg(row_w + c);
        const float sc = __half2float(__ldg(row_s + (c >> 1)));
        const uint4 x0 = __ldg(reinterpret_cast<const uint4*>(xm + c * 32));
        const uint4 x1 = __ldg(reinterpret_cast<const uint4*>(xm + c * 32 + 8));
        const uint4 x2 = __ldg(reinterpret_cast<const uint4*>(xm + c * 32 + 16));
        const uint4 x3 = __ldg(reinterpret_cast<const uint4*>(xm + c * 32 + 24));

        const unsigned ww[4] = {w.x, w.y, w.z, w.w};
        const uint4 xx[4] = {x0, x1, x2, x3};
        float part = 0.0f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const unsigned u = ww[p];
            const half2* h = reinterpret_cast<const half2*>(&xx[p]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(h[j]);
                const int q0 = static_cast<int>((u >> (8 * j)) & 15u) - kInt4Zero;
                const int q1 = static_cast<int>((u >> (8 * j + 4)) & 15u) - kInt4Zero;
                part = fmaf(static_cast<float>(q0), f.x, part);
                part = fmaf(static_cast<float>(q1), f.y, part);
            }
        }
        acc = fmaf(sc, part, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

// In-register 64-point FWHT
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

// Stage 1 for HSAL: Build global L2-resident LUT ONCE per token!
// For K=4096: 1-lift LUT = 512 * 128 * 2B = 128 KiB (3% of 4 MiB L2!)
//             2-lift LUT = 256 KiB (6% of 4 MiB L2!)
template <int Lifts>
__global__ void hsal_build_lut(const Half* __restrict__ codebooks, const Half* __restrict__ x,
                               Half* __restrict__ lut_out, int k) {
    // 1 block per 64-element FWHT segment (64 blocks for K=4096!), 128 threads/block
    // Each block computes 8 groups x 128 codes = 1024 LUT entries per lift!
    __shared__ float xh[64];
    const int seg = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;  // 0..127
    if (tid < 32) {
        const int base = m * k + seg * 64 + tid;
        float lo = __half2float(x[base]);
        float hi = __half2float(x[base + 32]);
        fwht64_pair(lo, hi, tid);
        xh[tid] = lo;
        xh[tid + 32] = hi;
    }
    __syncthreads();

    const int groups_total = k / kDim;
    const int c = tid;  // 0..127 (each thread owns one codebook entry!)
    #pragma unroll
    for (int lift = 0; lift < Lifts; ++lift) {
        float cb[8];
        #pragma unroll
        for (int d = 0; d < 8; ++d)
            cb[d] = __half2float(codebooks[(lift * kDim + d) * kCodes + c]);
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; ++d) dot = fmaf(cb[d], xh[g * 8 + d], dot);
            const int g_global = seg * 8 + g;
            lut_out[((m * Lifts + lift) * groups_total + g_global) * kCodes + c] = __float2half_rn(dot);
        }
    }
}

// Stage 2 for HSAL: ZERO SMEM, 100% Occupancy, coalesced uint4 streaming, L1/L2-cached LUT!
// Each lane reads 1 uint4 = 16 bytes = 128 weights (1 scale group!).
// Across 32 lanes = 512 bytes = 4096 weights in ONE SINGLE LOOP ITERATION!
template <int Lifts>
__global__ void hsal_stream_v2(const uint4* __restrict__ first, const uint4* __restrict__ second,
                               const Half* __restrict__ scale1, const Half* __restrict__ scale2,
                               const Half* __restrict__ lut, float* __restrict__ y, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;
    const int m = blockIdx.y;

    const int sgroups = k / kScaleGroup;  // 32 for K=4096 -> 1 iteration per warp!!
    const int groups_total = k / kDim;
    const Half* lut0 = lut + (m * Lifts) * groups_total * kCodes;
    const Half* lut1 = lut0 + groups_total * kCodes;

    float acc = 0.0f;
    for (int sg = lane; sg < sgroups; sg += 32) {
        const std::size_t pos = static_cast<std::size_t>(row) * sgroups + sg;
        const uint4 v0 = __ldg(first + pos);
        const float sc0 = __half2float(__ldg(scale1 + pos));
        const Half* lg0 = lut0 + sg * (16 * kCodes);
        const unsigned w0[4] = {v0.x, v0.y, v0.z, v0.w};
        float p0 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned u = w0[i];
            const unsigned c0 = u & 255u;
            const unsigned c1 = (u >> 8) & 255u;
            const unsigned c2 = (u >> 16) & 255u;
            const unsigned c3 = (u >> 24) & 255u;
            const float d0 = __half2float(__ldg(lg0 + (i * 4 + 0) * kCodes + (c0 & 127u)));
            const float d1 = __half2float(__ldg(lg0 + (i * 4 + 1) * kCodes + (c1 & 127u)));
            const float d2 = __half2float(__ldg(lg0 + (i * 4 + 2) * kCodes + (c2 & 127u)));
            const float d3 = __half2float(__ldg(lg0 + (i * 4 + 3) * kCodes + (c3 & 127u)));
            p0 += ((c0 & 128u) ? -d0 : d0) + ((c1 & 128u) ? -d1 : d1)
                + ((c2 & 128u) ? -d2 : d2) + ((c3 & 128u) ? -d3 : d3);
        }
        acc = fmaf(sc0, p0, acc);

        if constexpr (Lifts == 2) {
            const uint4 v1 = __ldg(second + pos);
            const float sc1 = __half2float(__ldg(scale2 + pos));
            const Half* lg1 = lut1 + sg * (16 * kCodes);
            const unsigned w1[4] = {v1.x, v1.y, v1.z, v1.w};
            float p1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const unsigned u = w1[i];
                const unsigned c0 = u & 255u;
                const unsigned c1 = (u >> 8) & 255u;
                const unsigned c2 = (u >> 16) & 255u;
                const unsigned c3 = (u >> 24) & 255u;
                const float d0 = __half2float(__ldg(lg1 + (i * 4 + 0) * kCodes + (c0 & 127u)));
                const float d1 = __half2float(__ldg(lg1 + (i * 4 + 1) * kCodes + (c1 & 127u)));
                const float d2 = __half2float(__ldg(lg1 + (i * 4 + 2) * kCodes + (c2 & 127u)));
                const float d3 = __half2float(__ldg(lg1 + (i * 4 + 3) * kCodes + (c3 & 127u)));
                p1 += ((c0 & 128u) ? -d0 : d0) + ((c1 & 128u) ? -d1 : d1)
                    + ((c2 & 128u) ? -d2 : d2) + ((c3 & 128u) ? -d3 : d3);
            }
            acc = fmaf(sc1, p1, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

// Sol Dual-Sign 2.5 bpw — Persistent 40-SM SMEM-LUT (builds 32 KiB LUT ONCE per SM, sweeps all rows!)
__global__ void sol_persistent_smem(const uint4* __restrict__ packed, const half2* __restrict__ scales,
                                    const Half* __restrict__ x, float* __restrict__ y, int n, int k) {
    extern __shared__ Half lut[];  // (k/4)*16 halfs = 32 KiB for K=4096
    const int m = blockIdx.y;
    const int total_lut = (k >> 2) * 16;
    for (int i = threadIdx.x; i < total_lut; i += blockDim.x) {
        const int g = i >> 4, mask = i & 15;
        const float a0 = __half2float(__ldg(x + m * k + g * 4 + 0));
        const float a1 = __half2float(__ldg(x + m * k + g * 4 + 1));
        const float a2 = __half2float(__ldg(x + m * k + g * 4 + 2));
        const float a3 = __half2float(__ldg(x + m * k + g * 4 + 3));
        float dot = ((mask & 1) ? -a0 : a0) + ((mask & 2) ? -a1 : a1)
                  + ((mask & 4) ? -a2 : a2) + ((mask & 8) ? -a3 : a3);
        lut[i] = __float2half_rn(dot);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_blk = blockDim.x >> 5;
    const int stride = gridDim.x * warps_per_blk;
    const int groups64 = k >> 6;

    for (int row = blockIdx.x * warps_per_blk + warp; row < n; row += stride) {
        float acc = 0.0f;
        for (int g = lane; g < groups64; g += 32) {
            const std::size_t pos = static_cast<std::size_t>(row) * groups64 + g;
            const uint4 v = __ldg(packed + pos);
            const unsigned words[4] = {v.x, v.y, v.z, v.w};
            float s1 = 0.0f, s2 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const unsigned u = words[w];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned code = (u >> (8 * b)) & 255u;
                    const int base = (g * 16 + w * 4 + b) * 16;
                    s1 += __half2float(lut[base + (code & 15u)]);
                    s2 += __half2float(lut[base + (code >> 4)]);
                }
            }
            const float2 sc = __half22float2(__ldg(scales + pos));
            acc = fmaf(sc.x, s1, acc);
            acc = fmaf(sc.y, s2, acc);
        }
        acc = warp_sum(acc);
        if (lane == 0) y[m * n + row] = acc;
    }
}

// Sol Dual-Sign 2.5 bpw — 2-Stage L2-Resident LUT (ZERO SMEM, 100% Occupancy!)
__global__ void sol_build_lut(const Half* __restrict__ x, Half* __restrict__ lut, int k) {
    const int m = blockIdx.y;
    const int total = (k >> 2) * 16;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int g = i >> 4, mask = i & 15;
    const float a0 = __half2float(x[m * k + g * 4 + 0]);
    const float a1 = __half2float(x[m * k + g * 4 + 1]);
    const float a2 = __half2float(x[m * k + g * 4 + 2]);
    const float a3 = __half2float(x[m * k + g * 4 + 3]);
    lut[m * total + i] = __float2half_rn(
        ((mask & 1) ? -a0 : a0) + ((mask & 2) ? -a1 : a1)
      + ((mask & 4) ? -a2 : a2) + ((mask & 8) ? -a3 : a3));
}

__global__ void sol_stream_l2(const uint4* __restrict__ packed, const half2* __restrict__ scales,
                              const Half* __restrict__ lut_all, float* __restrict__ y, int n, int k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarps + warp;
    if (row >= n) return;
    const int m = blockIdx.y;
    const Half* lut = lut_all + m * ((k >> 2) * 16);
    const int groups64 = k >> 6;
    float acc = 0.0f;
    for (int g = lane; g < groups64; g += 32) {
        const std::size_t pos = static_cast<std::size_t>(row) * groups64 + g;
        const uint4 v = __ldg(packed + pos);
        const unsigned words[4] = {v.x, v.y, v.z, v.w};
        float s1 = 0.0f, s2 = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            const unsigned u = words[w];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned code = (u >> (8 * b)) & 255u;
                const int base = (g * 16 + w * 4 + b) * 16;
                s1 += __half2float(__ldg(lut + base + (code & 15u)));
                s2 += __half2float(__ldg(lut + base + (code >> 4)));
            }
        }
        const float2 sc = __half22float2(__ldg(scales + pos));
        acc = fmaf(sc.x, s1, acc);
        acc = fmaf(sc.y, s2, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[m * n + row] = acc;
}

// --- Host reference & encoding (identical to v1!) ---
void host_fwht(float* values, int size) {
    for (int start = 0; start < size; start += 64) {
        for (int width = 1; width < 64; width *= 2) {
            for (int b = start; b < start + 64; b += 2 * width) {
                for (int i = b; i < b + width; ++i) {
                    const float a = values[i], c = values[i + width];
                    values[i] = a + c;
                    values[i + width] = a - c;
                }
            }
        }
        for (int i = start; i < start + 64; ++i) values[i] *= 0.125f;
    }
}

using Codebook = std::array<std::array<float, kDim>, kCodes>;
Codebook make_codebook() {
    Codebook book{};
    const float root = __half2float(__float2half_rn(std::sqrt(2.0f)));
    int index = 0;
    for (int i = 0; i < kDim; ++i)
        for (int j = i + 1; j < kDim; ++j) {
            book[index][i] = root; book[index++][j] = root;
            book[index][i] = root; book[index++][j] = -root;
        }
    for (int mask = 0; mask < 64; ++mask) {
        int parity = 0;
        book[index][0] = root * 0.5f;
        for (int d = 1; d < 7; ++d) {
            const int neg = (mask >> (d - 1)) & 1;
            parity ^= neg;
            book[index][d] = neg ? -root * 0.5f : root * 0.5f;
        }
        book[index++][7] = parity ? -root * 0.5f : root * 0.5f;
    }
    for (int i = 0; i < kDim; ++i) book[index++][i] = 2.0f * root;
    return book;
}

Byte nearest_code(const float* x) {
    int largest = 0, second = 1, smallest = 0, sign_mask = 0, parity = 0;
    if (std::abs(x[second]) > std::abs(x[largest])) std::swap(largest, second);
    float l1 = 0.0f;
    for (int d = 0; d < kDim; ++d) {
        const float a = std::abs(x[d]); l1 += a;
        if (d >= 2 && a > std::abs(x[largest])) { second = largest; largest = d; }
        else if (d >= 2 && a > std::abs(x[second])) { second = d; }
        if (a < std::abs(x[smallest])) smallest = d;
        if (x[d] < 0.0f) { sign_mask |= 1 << d; parity ^= 1; }
    }
    const int i = std::min(largest, second), j = std::max(largest, second);
    const int si = (sign_mask >> i) & 1, sj = (sign_mask >> j) & 1;
    const float pair_dot = std::abs(x[i]) + std::abs(x[j]);
    const float axis_dot = 2.0f * std::abs(x[largest]);
    const float half_dot = 0.5f * l1 - (parity ? std::abs(x[smallest]) : 0.0f);
    if (parity) sign_mask ^= 1 << smallest;
    const int sign = sign_mask & 1;
    if (sign) sign_mask ^= 255;
    if (half_dot >= std::max(pair_dot, axis_dot))
        return static_cast<Byte>((56 + ((sign_mask >> 1) & 63)) | (sign << 7));
    if (axis_dot >= pair_dot)
        return static_cast<Byte>((120 + largest) | ((x[largest] < 0.0f) ? 128 : 0));
    const int pair = i * (15 - i) / 2 + j - i - 1;
    return static_cast<Byte>((2 * pair + (si ^ sj)) | (si << 7));
}

float coeff(const Codebook& book, Byte code, int d) {
    const float v = book[code & 127][d];
    return (code & 128) ? -v : v;
}

struct Weights {
    std::vector<Half> fp16, int4_scale, first_scale, second_scale, books;
    std::vector<Byte> int4, first, second, sol_codes;
    std::vector<half2> sol_scales;
    Codebook book = make_codebook();

    Weights(int n, int k, std::mt19937& rng)
        : fp16(n * k), int4_scale(n * k / 64),
          first_scale(n * k / kScaleGroup), second_scale(first_scale.size()),
          books(2 * kCodes * kDim), int4(n * k / 2), first(n * k / kDim),
          second(first.size()), sol_codes(n * k / 4), sol_scales(n * k / 64) {
        std::normal_distribution<float> normal(0.0f, 0.025f);
        for (Half& w : fp16) w = __float2half_rn(normal(rng));
        for (int c = 0; c < kCodes; ++c)
            for (int d = 0; d < kDim; ++d)
                books[kDim * kCodes + d * kCodes + c] = books[d * kCodes + c] = __float2half_rn(book[c][d]);
        std::vector<float> rot(k);
        for (int r = 0; r < n; ++r) {
            const int base = r * k;
            for (int i = 0; i < k; ++i) rot[i] = __half2float(fp16[base + i]);
            for (int i = 0; i < k; i += 64) encode_scalar(rot.data() + i, base + i);
            host_fwht(rot.data(), k);
            for (int i = 0; i < k; i += kScaleGroup) {
                encode_lift(rot.data() + i, base + i, first, first_scale);
                encode_lift(rot.data() + i, base + i, second, second_scale);
            }
        }
    }
    void encode_scalar(const float* w, int offset) {
        float mag = 0.0f, mean = 0.0f;
        for (int i = 0; i < 64; ++i) { mag = std::max(mag, std::abs(w[i])); mean += std::abs(w[i]); }
        const int g = offset / 64;
        int4_scale[g] = __float2half_rn(mag / 7.0f);
        const float sc = __half2float(int4_scale[g]);
        const Half a1 = __float2half_rn(mean / 64.0f);
        float res[64], rmean = 0.0f;
        for (int i = 0; i < 64; ++i) {
            const int q = sc == 0.0f ? 0 : std::clamp(static_cast<int>(std::nearbyint(w[i] / sc)), -8, 7);
            int4[(offset + i) / 2] |= static_cast<Byte>((q + kInt4Zero) << (4 * (i % 2)));
            const float a = __half2float(a1);
            res[i] = w[i] - (w[i] < 0.0f ? -a : a);
            rmean += std::abs(res[i]);
            const int s = (w[i] < 0.0f ? 1 : 0) | (res[i] < 0.0f ? 16 : 0);
            sol_codes[(offset + i) / 4] |= static_cast<Byte>(s << (i % 4));
        }
        sol_scales[g] = __halves2half2(a1, __float2half_rn(rmean / 64.0f));
    }
    void encode_lift(float* w, int offset, std::vector<Byte>& codes, std::vector<Half>& scales) {
        float norm = 0.0f;
        for (int i = 0; i < kScaleGroup; ++i) norm += w[i] * w[i];
        const Half sc = __float2half_rn(std::sqrt(norm / kScaleGroup));
        scales[offset / kScaleGroup] = sc;
        for (int g = 0; g < kScaleGroup; g += kDim) {
            const Byte code = nearest_code(w + g);
            codes[(offset + g) / kDim] = code;
            for (int d = 0; d < kDim; ++d) w[g + d] -= __half2float(sc) * coeff(book, code, d);
        }
    }
};

enum class Method { Fp16, Int4v2, Hsal1v2, Hsal2v2, SolPersist, SolL2 };
const char* mname(Method m) {
    switch (m) {
        case Method::Fp16: return "cublas_fp16";
        case Method::Int4v2: return "packed_int4_v2";
        case Method::Hsal1v2: return "hsal_1lift_l2lut";
        case Method::Hsal2v2: return "hsal_2lift_l2lut";
        case Method::SolPersist: return "sol_2.5b_persist_smem";
        case Method::SolL2: return "sol_2.5b_l2lut";
    }
    return "";
}

float dec_w(const Weights& w, Method m, int pos) {
    if (m == Method::Fp16) return __half2float(w.fp16[pos]);
    if (m == Method::Int4v2) {
        int q = ((w.int4[pos / 2] >> (4 * (pos % 2))) & 15) - kInt4Zero;
        return q * __half2float(w.int4_scale[pos / 64]);
    }
    if (m == Method::Hsal1v2 || m == Method::Hsal2v2) {
        int g = pos / kScaleGroup, d = pos % kDim;
        float v = __half2float(w.first_scale[g]) * coeff(w.book, w.first[pos / kDim], d);
        if (m == Method::Hsal2v2) v += __half2float(w.second_scale[g]) * coeff(w.book, w.second[pos / kDim], d);
        return v;
    }
    float2 sc = __half22float2(w.sol_scales[pos / 64]);
    int code = w.sol_codes[pos / 4] >> (pos % 4);
    return ((code & 1) ? -sc.x : sc.x) + ((code & 16) ? -sc.y : sc.y);
}

struct Metrics { float max_err, cos; };
Metrics check_acc(const Weights& w, const std::vector<Half>& in, const std::vector<float>& out,
                  Method m, int n, int k, int batch) {
    std::vector<float> xv(k);
    float max_e = 0.0f, min_cos = 1.0f;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < k; ++i) xv[i] = __half2float(in[b * k + i]);
        if (m == Method::Hsal1v2 || m == Method::Hsal2v2) host_fwht(xv.data(), k);
        double dot = 0, nr = 0, no = 0;
        for (int r = 0; r < n; ++r) {
            double ref = 0;
            for (int i = 0; i < k; ++i) ref += static_cast<double>(dec_w(w, m, r * k + i)) * xv[i];
            float got = out[b * n + r];
            max_e = std::max(max_e, std::abs(got - static_cast<float>(ref)));
            dot += ref * got; nr += ref * ref; no += static_cast<double>(got) * got;
        }
        min_cos = std::min(min_cos, static_cast<float>(dot / (std::sqrt(nr) * std::sqrt(no))));
    }
    return {max_e, min_cos};
}

template <class Fn>
float time_graph(cudaStream_t stream, Fn&& launch, std::uint32_t* flush, int sms, bool cold) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    for (int i = 0; i < kWarmups; ++i) {
        if (cold) flush_l2<<<sms * 4, 256, 0, stream>>>(flush, kFlushBytes / 4);
        CHECK(cudaGraphLaunch(exec, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    std::vector<float> times(kSamples);
    for (int i = 0; i < kSamples; ++i) {
        if (cold) flush_l2<<<sms * 4, 256, 0, stream>>>(flush, kFlushBytes / 4);
        CHECK(cudaEventRecord(s, stream));
        CHECK(cudaGraphLaunch(exec, stream));
        CHECK(cudaEventRecord(e, stream));
        CHECK(cudaEventSynchronize(e));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, s, e));
        times[i] = ms * 1000.0f;
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    std::sort(times.begin(), times.end());
    return times[kSamples / 2];
}

void run_shape(Context& ctx, const Weights& w, int n, int k, int M, int sms, std::uint32_t* flush, std::mt19937& rng, bool& first_out) {
    std::normal_distribution<float> normal;
    std::vector<Half> input(M * k);
    for (Half& a : input) a = __float2half_rn(normal(rng));
    const DeviceBuffer<Half> x(input), fp16(w.fp16), int4_scale(w.int4_scale),
        scale1(w.first_scale), scale2(w.second_scale), books(w.books);
    const DeviceBuffer<Byte> int4(w.int4), first(w.first), second(w.second), sol_codes(w.sol_codes);
    const DeviceBuffer<half2> sol_scale(w.sol_scales);
    const DeviceBuffer<Half> hsal_lut(M * 2 * (k / kDim) * kCodes);
    const DeviceBuffer<Half> sol_lut(M * (k / 4) * 16);
    const DeviceBuffer<float> y(M * n);
    std::vector<float> out(M * n);

    float base_cold = 0, base_warm = 0;
    for (Method m : {Method::Fp16, Method::Int4v2, Method::Hsal1v2, Method::Hsal2v2, Method::SolPersist, Method::SolL2}) {
        auto launch = [&] {
            if (m == Method::Fp16) {
                const float alpha = 1.0f, beta = 0.0f;
                CHECK(cublasGemmEx(ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, M, k,
                    &alpha, fp16.get(), CUDA_R_16F, k, x.get(), CUDA_R_16F, k,
                    &beta, y.get(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else if (m == Method::Int4v2) {
                packed_int4_v2<<<dim3((n + kWarps - 1) / kWarps, M), kThreads, 0, ctx.stream>>>(
                    reinterpret_cast<const uint4*>(int4.get()), int4_scale.get(), x.get(), y.get(), n, k);
            } else if (m == Method::Hsal1v2) {
                hsal_build_lut<1><<<dim3(k / 64, M), 128, 0, ctx.stream>>>(books.get(), x.get(), hsal_lut.get(), k);
                hsal_stream_v2<1><<<dim3((n + kWarps - 1) / kWarps, M), kThreads, 0, ctx.stream>>>(
                    reinterpret_cast<const uint4*>(first.get()), nullptr, scale1.get(), nullptr, hsal_lut.get(), y.get(), n, k);
            } else if (m == Method::Hsal2v2) {
                hsal_build_lut<2><<<dim3(k / 64, M), 128, 0, ctx.stream>>>(books.get(), x.get(), hsal_lut.get(), k);
                hsal_stream_v2<2><<<dim3((n + kWarps - 1) / kWarps, M), kThreads, 0, ctx.stream>>>(
                    reinterpret_cast<const uint4*>(first.get()), reinterpret_cast<const uint4*>(second.get()),
                    scale1.get(), scale2.get(), hsal_lut.get(), y.get(), n, k);
            } else if (m == Method::SolPersist) {
                sol_persistent_smem<<<dim3(sms, M), 256, (k / 4) * 16 * sizeof(Half), ctx.stream>>>(
                    reinterpret_cast<const uint4*>(sol_codes.get()), sol_scale.get(), x.get(), y.get(), n, k);
            } else if (m == Method::SolL2) {
                int tot = (k / 4) * 16;
                sol_build_lut<<<dim3((tot + 255) / 256, M), 256, 0, ctx.stream>>>(x.get(), sol_lut.get(), k);
                sol_stream_l2<<<dim3((n + kWarps - 1) / kWarps, M), kThreads, 0, ctx.stream>>>(
                    reinterpret_cast<const uint4*>(sol_codes.get()), sol_scale.get(), sol_lut.get(), y.get(), n, k);
            }
            CHECK(cudaGetLastError());
        };
        float cold = time_graph(ctx.stream, launch, flush, sms, true);
        float warm = time_graph(ctx.stream, launch, flush, sms, false);
        if (m == Method::Fp16) { base_cold = cold; base_warm = warm; }
        launch();
        CHECK(cudaStreamSynchronize(ctx.stream));
        CHECK(cudaMemcpy(out.data(), y.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Metrics acc = check_acc(w, input, out, m, n, k, M);

        std::size_t bytes = 0;
        if (m == Method::Fp16) bytes = w.fp16.size() * 2;
        else if (m == Method::Int4v2) bytes = w.int4.size() + w.int4_scale.size() * 2;
        else if (m == Method::Hsal1v2) bytes = w.first.size() + w.first_scale.size() * 2;
        else if (m == Method::Hsal2v2) bytes = w.first.size() + w.second.size() + (w.first_scale.size() + w.second_scale.size()) * 2;
        else bytes = w.sol_codes.size() + w.sol_scales.size() * 4;

        if (!first_out) std::cout << ",\n";
        first_out = false;
        std::cout << "    {\"method\":\"" << mname(m) << "\",\"m\":" << M << ",\"n\":" << n << ",\"k\":" << k
                  << ",\"bytes\":" << bytes << ",\"bpw\":" << (8.0 * bytes / (n * k))
                  << ",\"cold_us\":" << std::fixed << std::setprecision(2) << cold
                  << ",\"cold_gbps\":" << std::setprecision(1) << (bytes / (1000.0 * cold))
                  << ",\"cold_speedup\":" << std::setprecision(2) << (base_cold / cold)
                  << ",\"warm_us\":" << std::setprecision(2) << warm
                  << ",\"warm_gbps\":" << std::setprecision(1) << (bytes / (1000.0 * warm))
                  << ",\"warm_speedup\":" << std::setprecision(2) << (base_warm / warm)
                  << ",\"cos\":" << std::setprecision(7) << acc.cos << "}";
    }
}
}  // namespace

int main() {
    cudaDeviceProp dev{};
    CHECK(cudaGetDeviceProperties(&dev, 0));
    Context ctx;
    DeviceBuffer<std::uint32_t> flush(kFlushBytes / 4);
    CHECK(cudaMemsetAsync(flush.get(), 0, kFlushBytes, ctx.stream));
    CHECK(cudaStreamSynchronize(ctx.stream));
    std::mt19937 rng(574103);
    std::cout << "{\n  \"gpu\":\"" << dev.name << "\",\"sms\":" << dev.multiProcessorCount << ",\n  \"results\":[\n";
    bool first_out = true;
    for (auto [n, k] : {std::pair{4096, 4096}, std::pair{3072, 1024}}) {
        Weights w(n, k, rng);
        run_shape(ctx, w, n, k, 1, dev.multiProcessorCount, flush.get(), rng, first_out);
        run_shape(ctx, w, n, k, 4, dev.multiProcessorCount, flush.get(), rng, first_out);
    }
    std::cout << "\n  ]\n}\n";
    return 0;
}
