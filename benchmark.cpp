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

constexpr int N = 64, M = 64, K = 64;
class GemmBenchmark {
public:
    using GemmFunc = std::function<void(uint8_t*, uint8_t*, int*, int, int, int)>;

    GemmBenchmark(GemmFunc func)
        : gemm(func)
    {
        A = genTNNMatrix(N, K);
        B = genBNNMatrix(K, M);
        res.resize(N * M);

        Anew.resize((N * K + 3) / 4);
        Bnew.resize((K * M + 7) / 8);
        encoder::encodeTern(A.get(), Anew.data(), N, K);
        encoder::encodeBin(B.get(), Bnew.data(), K, M);
    }

    void run(benchmark::State& state) {
        for (auto _ : state) {
            gemm(Anew.data(), Bnew.data(), res.data(), N, M, K);
            benchmark::DoNotOptimize(res);
        }
    }

private:
    GemmFunc gemm;
    std::unique_ptr<int[]> A;
    std::unique_ptr<int[]> B;
    std::vector<uint8_t> Anew;
    std::vector<uint8_t> Bnew;
    std::vector<int> res;
};

static void BM_GemmV0(benchmark::State& state) {
    std::unique_ptr<int[]> A = genTNNMatrix(N, K);
    std::unique_ptr<int[]> B = genBNNMatrix(K, M);
    std::vector<int> res(N * M);
    for (auto _ : state) {
        gemmV0(A.get(), B.get(), res.data(), N, M, K);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_GemmV1(benchmark::State& state) {
    static GemmBenchmark bm(gemmV1);
    bm.run(state);
}

BENCHMARK(BM_GemmV0);
BENCHMARK(BM_GemmV1);

BENCHMARK_MAIN();