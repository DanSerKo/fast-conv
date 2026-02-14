#include "gemm/gemms.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

TEST(Correct_Test, V1) {
    const int N = 64, M = 64, K = 64;
    std::vector<int> A_orig(N * K);
    std::vector<int> B_orig(K * M);
    std::vector<uint8_t> Anew((N * K + 3) / 4), Bnew((K * M + 7) / 8);
    
    std::vector<int> resV0(N * M);
    std::vector<int> resV1(N * M);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> distTernary(-1, 1);
    std::uniform_int_distribution<> distBinary(0, 1);

    for (int i = 0; i < N * K; i++)  {
        A_orig[i] = distTernary(gen);
    }
    for (int i = 0; i < K * M; i++) {
        B_orig[i] = distBinary(gen) ? 1 : -1;
    }
    
    encoder::encodeTern(A_orig.data(), Anew.data(), N, K);
    encoder::encodeBinT(B_orig.data(), Bnew.data(), K, M);
    gemmV0(A_orig.data(), B_orig.data(), resV0.data(), N, M, K);
    gemmV1(Anew.data(), Bnew.data(), resV1.data(), N, M, K);
    for (int i = 0; i < N * M; i++) {
        ASSERT_EQ(resV0[i], resV1[i]) << "Ошибка в ячейке [" << i / M << "][" << i % M << "]";
    }
}

TEST(Correct_Test, V2) {
    const int N = 64, M = 64, K = 64;
    std::vector<int> A_orig(N * K);
    std::vector<int> B_orig(K * M);
    std::vector<uint8_t> Anew((N * K + 3) / 4), Bnew((K * M + 7) / 8);
    
    std::vector<int> resV0(N * M);
    std::vector<int> resV1(N * M);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> distTernary(-1, 1);
    std::uniform_int_distribution<> distBinary(0, 1);

    for (int i = 0; i < N * K; i++)  {
        A_orig[i] = distTernary(gen);
    }
    for (int i = 0; i < K * M; i++) {
        B_orig[i] = distBinary(gen) ? 1 : -1;
    }
    
    encoder::encodeTern(A_orig.data(), Anew.data(), N, K);
    encoder::encodeBinT(B_orig.data(), Bnew.data(), K, M);
    gemmV0(A_orig.data(), B_orig.data(), resV0.data(), N, M, K);
    gemmV2(Anew.data(), Bnew.data(), resV1.data(), N, M, K);
    for (int i = 0; i < N * M; i++) {
        ASSERT_EQ(resV0[i], resV1[i]) << "Ошибка в ячейке [" << i / M << "][" << i % M << "]";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}