#include "gemm/gemms.h"

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

class GemmFixture {
public:
    std::vector<uint8_t> Anew, Bnew;
    std::vector<int> res;

    GemmFixture(int n, int m, int k) {
        auto A_raw = genTNNMatrix(n, k);
        auto B_raw = genBNNMatrix(k, m);
        res.resize(n * m);

        Anew.resize((n * k + 3) / 4);
        Bnew.resize((k * m + 7) / 8);
        encoder::encodeTern(A_raw.get(), Anew.data(), n, k);
        encoder::encodeBin(B_raw.get(), Bnew.data(), k, m);
    }
};

static void BM_GemmV0(benchmark::State& state) {
    const int n = state.range(0);
    const int m = state.range(1);
    const int k = state.range(2);

    auto A = genTNNMatrix(n, k);
    auto B = genBNNMatrix(k, m);
    std::vector<int> res(n * m);

    for (auto _ : state) {
        gemmV0(A.get(), B.get(), res.data(), n, m, k);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_GemmV1(benchmark::State& state) {
    const int n = state.range(0);
    const int m = state.range(1);
    const int k = state.range(2);
    
    GemmFixture fix(n, m, k);

    for (auto _ : state) {
        gemmV1(fix.Anew.data(), fix.Bnew.data(), fix.res.data(), n, m, k);
        benchmark::DoNotOptimize(fix.res);
    }
}

#define SET_ARGS \
    Args({32, 1024, 9})       /* (32x32, 1ch) */ \
    ->Args({64, 16384, 27})    /* (128x128, 3ch) */ \
    ->Args({128, 65536, 54})   /* (256x256, 6ch) */ \
    ->Args({64, 65536, 288})  /* (256x256, 6ch, 32) */ \
    ->Args({16, 262144, 27})   /* (512x512, 3) */ \
    ->Args({256, 16384, 2304}) /* (128x128, 256ch) */ \
    ->Args({512, 4096, 2304})  /* (64x64, 256ch) */ \
    ->Args({256, 256, 4608})   /* (16x16, 512ch) */ \
    ->Unit(benchmark::kMicrosecond) \
    ->Repetitions(5) \
    ->DisplayAggregatesOnly(true)


BENCHMARK(BM_GemmV0)->SET_ARGS;
BENCHMARK(BM_GemmV1)->SET_ARGS;

BENCHMARK_MAIN();