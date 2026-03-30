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

TEST(Correct_Test, V3) {
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
    gemmV3(Anew.data(), Bnew.data(), resV1.data(), N, M, K);
    for (int i = 0; i < N * M; i++) {
        ASSERT_EQ(resV0[i], resV1[i]) << "Ошибка в ячейке [" << i / M << "][" << i % M << "]";
    }
}

void run_correctness_test(void(*gemm_func)(uint8_t*, uint8_t*, int*, int, int, int), const std::string& name) {
    const int N = 64, M = 64, K = 256;
    std::vector<int> A_orig(N * K);
    std::vector<int> B_orig(K * M);
    std::vector<uint8_t> Anew((N * K + 3) / 4 * 2), Bnew((K * M + 7) / 8); 
    
    std::vector<int> resV0(N * M);
    std::vector<int> resTest(N * M);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> distTernary(-1, 1);
    std::uniform_int_distribution<> distBinary(0, 1);

    for (int i = 0; i < N * K; i++)  A_orig[i] = distTernary(gen);
    for (int i = 0; i < K * M; i++) B_orig[i] = distBinary(gen) ? 1 : -1;
    
    encoder::encodeTern(A_orig.data(), Anew.data(), N, K);
    encoder::encodeBinT(B_orig.data(), Bnew.data(), K, M);
    
    gemmV0(A_orig.data(), B_orig.data(), resV0.data(), N, M, K);
    gemm_func(Anew.data(), Bnew.data(), resTest.data(), N, M, K);
    
    for (int i = 0; i < N * M; i++) {
        ASSERT_EQ(resV0[i], resTest[i]) << "Ошибка в " << name << " в ячейке [" << i / M << "][" << i % M << "]";
    }
}

TEST(Correct_Test, V5_Blocked) { run_correctness_test(gemmV5_blocked, "V5_Blocked"); }
TEST(Correct_Test, V6_Packed) { run_correctness_test(gemmV6_packed, "V6_Packed"); }
TEST(Correct_Test, V7_SSE) { run_correctness_test(gemmV7_SSE, "V7_SSE"); }
TEST(Correct_Test, V8_AVX) { run_correctness_test(gemmV8_AVX, "V8_AVX"); }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}