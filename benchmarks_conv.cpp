#include "im2col/conv.h"
#include "gemm/gemms.h"
#include "util/encoder.h"

#include <benchmark/benchmark.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

std::unique_ptr<int[]> genTNNMatrix(int n, int m) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist(-1, 1);
    std::unique_ptr<int[]> matrix = std::make_unique<int[]>(n * m); 
    for (int i = 0; i < n * m; i++) {
        matrix[i] = dist(gen);
    }
    return matrix;
}

std::unique_ptr<int[]> genBNNMatrix(int n, int m) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> dist(0, 1);
    std::unique_ptr<int[]> matrix = std::make_unique<int[]>(n * m); 
    for (int i = 0; i < n * m; i++) {
        matrix[i] = dist(gen) ? 1 : -1; 
    }
    return matrix;
}

class ConvFixture {
public:
    std::vector<uint8_t> Anew;
    std::vector<int> image;
    std::vector<int> weights;
    std::vector<int> output;
    int K_padded;

    ConvFixture(int H, int W, int C_in, int C_out, int K_h, int K_w, int stride, int pad) {
        int H_out = (H + 2 * pad - K_h) / stride + 1;
        int W_out = (W + 2 * pad - K_w) / stride + 1;
        int M = H_out * W_out;
        
        int K_actual = C_in * K_h * K_w;
        K_padded = ((K_actual + 7) / 8) * 8;

        auto img_raw = genBNNMatrix(C_in * H, W);
        image.assign(img_raw.get(), img_raw.get() + C_in * H * W);

        auto weights_raw = genTNNMatrix(C_out, K_padded);
        weights.assign(weights_raw.get(), weights_raw.get() + C_out * K_padded);

        Anew.resize((C_out * K_padded + 3) / 4);
        encoder::encodeTern(weights_raw.get(), Anew.data(), C_out, K_padded);

        output.resize(C_out * M);
    }
};

static void BM_Conv2d_Naive(benchmark::State& state) {
    int H = state.range(0);
    int W = state.range(1);
    int C_in = state.range(2);
    int C_out = state.range(3);
    int K_h = state.range(4);
    int K_w = state.range(5);
    int stride = 1; 
    int pad = 1;

    ConvFixture fix(H, W, C_in, C_out, K_h, K_w, stride, pad);

    for (auto _ : state) {
        conv2d_naive(
            fix.image.data(), fix.weights.data(), fix.output.data(),
            C_in, C_out, H, W, K_h, K_w, stride, stride, pad, pad, fix.K_padded, 1
        );
        benchmark::DoNotOptimize(fix.output);
    }
}

static void BM_Conv_Packed(benchmark::State& state, gemm_func_t gemm_impl) {
    int H = state.range(0);
    int W = state.range(1);
    int C_in = state.range(2);
    int C_out = state.range(3);
    int K_h = state.range(4);
    int K_w = state.range(5);
    int P = state.range(6);
    int stride = 1;
    int pad = 1;

    ConvFixture fix(H, W, C_in, C_out, K_h, K_w, stride, pad);

    for (auto _ : state) {
        p_im2col_packed_conv(
            fix.image.data(), fix.Anew.data(), fix.output.data(),
            C_in, H, W, C_out, K_h, K_w, stride, pad, P, gemm_impl
        );
        benchmark::DoNotOptimize(fix.output);
    }
}

static void BM_Conv_Packed_Threads(benchmark::State& state, gemm_func_t gemm_impl) {
    int H = state.range(0);
    int W = state.range(1);
    int C_in = state.range(2);
    int C_out = state.range(3);
    int K_h = state.range(4);
    int K_w = state.range(5);
    int P = state.range(6);
    int num_threads = state.range(7);
    int stride = 1;
    int pad = 1;

    ConvFixture fix(H, W, C_in, C_out, K_h, K_w, stride, pad);

    for (auto _ : state) {
        p_im2col_packed_conv_threads(
            fix.image.data(), fix.Anew.data(), fix.output.data(),
            C_in, H, W, C_out, K_h, K_w, stride, pad, P, num_threads, gemm_impl
        );
        benchmark::DoNotOptimize(fix.output);
    }
}

#define SET_NAIVE_ARGS \
    Args({32, 32, 1, 32, 3, 3}) \
    ->Args({64, 64, 3, 64, 3, 3}) \
    ->Args({128, 128, 3, 64, 3, 3}) \
    ->Args({256, 256, 6, 128, 64, 3, 3}) \
    ->Unit(benchmark::kMicrosecond) \
    ->Repetitions(10) \
    ->DisplayAggregatesOnly(true)

#define SET_PACKED_ARGS \
    Args({32, 32, 1, 32, 3, 3, 64}) \
    ->Args({32, 32, 1, 32, 3, 3, 128}) \
    ->Args({128, 128, 3, 64, 3, 3, 128}) \
    ->Args({128, 128, 3, 64, 3, 3, 256}) \
    ->Args({256, 256, 6, 128, 3, 3, 128}) \
    ->Unit(benchmark::kMicrosecond) \
    ->Repetitions(10) \
    ->DisplayAggregatesOnly(true)

#define SET_THREADS_ARGS \
    Args({128, 128, 3, 64, 3, 3, 128, 2}) \
    ->Args({128, 128, 3, 64, 3, 3, 128, 4}) \
    ->Args({128, 128, 3, 64, 3, 3, 128, 8}) \
    ->Args({256, 256, 6, 128, 3, 3, 128, 4}) \
    ->Args({256, 256, 6, 128, 3, 3, 128, 8}) \
    ->Unit(benchmark::kMicrosecond) \
    ->Repetitions(10) \
    ->DisplayAggregatesOnly(true)

BENCHMARK(BM_Conv2d_Naive)->SET_NAIVE_ARGS;

BENCHMARK([](benchmark::State& state){ BM_Conv_Packed(state, gemmV12_AVX_4x32); })->SET_PACKED_ARGS;
BENCHMARK([](benchmark::State& state){ BM_Conv_Packed(state, gemmV14_BLIS_SingleThread); })->SET_PACKED_ARGS;
BENCHMARK([](benchmark::State& state){ BM_Conv_Packed(state, gemmCandidate); })->SET_PACKED_ARGS;

BENCHMARK([](benchmark::State& state){ BM_Conv_Packed_Threads(state, gemmV12_AVX_4x32); })->SET_THREADS_ARGS;
BENCHMARK([](benchmark::State& state){ BM_Conv_Packed_Threads(state, gemmV14_BLIS_SingleThread); })->SET_THREADS_ARGS;
BENCHMARK([](benchmark::State& state){ BM_Conv_Packed_Threads(state, gemmCandidate); })->SET_THREADS_ARGS;

BENCHMARK_MAIN();