#include "gemm/gemms.h"
#include "im2col/conv.h"

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
//TEST(Correct_Test, V9_AVX_Popcnt) { run_correctness_test(gemmV9_AVX_FastPopcnt, "V9"); }
//TEST(Correct_Test, V10_AVX_Unrolled) { run_correctness_test(gemmV10_AVX_Unrolled, "V10"); }
TEST(Correct_Test, V12_AVX_4x32_Microkernel) { run_correctness_test(gemmV12_AVX_4x32, "V12"); }
TEST(Correct_Test, V14_BLIS_Single) { run_correctness_test(gemmV14_BLIS_SingleThread, "V14"); }
TEST(Correct_Test, V16_BLIS_CorrectOrder) { run_correctness_test(gemmV16_BLIS_CorrectOrder, "V16"); }
TEST(Correct_Test, V17_BLIS_FastKernel) { run_correctness_test(gemmV17_BLIS_FastKernel, "V17"); }
TEST(Correct_Test, V18_Ultimate_SingleThread) { run_correctness_test(gemmV18_Ultimate_SingleThread, "V18"); }
TEST(Correct_Test, V14_BLIS_SingleThread_Optimized) { run_correctness_test(gemmV14_BLIS_SingleThread_Optimized, "V14opt"); }
TEST(Correct_Test, V19_TheUltimate_ST) { run_correctness_test(gemmV19_TheUltimate_ST, "V19"); }
TEST(Correct_Test, gemmCandidate) { run_correctness_test(gemmCandidate, "Candidate"); }



void run_conv_correctness_test(
    void(*gemm_func)(uint8_t*, uint8_t*, int*, int, int, int), 
    const std::string& name
) {
    const int C_in = 16, H = 14, W = 14; 
    const int C_out = 32, K_h = 3, K_w = 3;
    const int stride_h = 1, stride_w = 1;
    const int pad_h = 1, pad_w = 1;

    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    int M = H_out * W_out; 
    
    int K_actual = C_in * K_h * K_w;
    int K_padded = ((K_actual + 7) / 8) * 8;
    int N = C_out;

    std::vector<int> image_orig(C_in * H * W);
    std::vector<int> weights_orig(N * K_padded, 0);
    
    std::vector<uint8_t> Anew((N * K_padded) / 4, 0); 
    std::vector<uint8_t> Bnew((K_padded / 8) * M, 0); 
    
    std::vector<int> res_naive(N * M, 0);
    std::vector<int> res_gemm(N * M, 0);

    std::mt19937 gen(1234);
    std::uniform_int_distribution<> distTernary(-1, 1);
    std::uniform_int_distribution<> distBinary(0, 1);

    for (int i = 0; i < C_in * H * W; i++) {
        image_orig[i] = distBinary(gen) ? 1 : -1;
    }
    
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K_actual; k++) {
            weights_orig[n * K_padded + k] = distTernary(gen);
        }
    }

    conv2d_naive(
        image_orig.data(), weights_orig.data(), res_naive.data(), 
        C_in, C_out, H, W, K_h, K_w, stride_h, stride_w, pad_h, pad_w, K_padded
    );

    encoder::encodeTern(weights_orig.data(), Anew.data(), N, K_padded);
    
    im2col_packed_bin(
        image_orig.data(), Bnew.data(), 
        C_in, H, W, K_h, K_w, stride_h, stride_w, pad_h, pad_w
    );
    
    gemm_func(Anew.data(), Bnew.data(), res_gemm.data(), N, M, K_padded);

    for (int i = 0; i < N * M; i++) {
        ASSERT_EQ(res_naive[i], res_gemm[i]) 
            << "Ошибка в " << name 
            << " | Канал: " << i / M 
            << " Пространственная позиция: " << i % M;
    }
}

TEST(Conv_Correct_Test, V14_BLIS_Optimized_Conv) { 
    run_conv_correctness_test(gemmV14_BLIS_SingleThread_Optimized, "V14_Conv"); 
}

TEST(Conv_Correct_Test, P_Im2Col_Binary_Ternary) {
    const int C_in = 8;
    const int H = 10, W = 10;
    const int C_out = 16;
    const int K_h = 3, K_w = 3;
    const int stride = 1, pad = 1;
    const int P = 16;

    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;
    int M = H_out * W_out;
    int K_actual = C_in * K_h * K_w;
    int K_padded = ((K_actual + 7) / 8) * 8;

    std::vector<int> image(C_in * H * W);
    std::vector<int> weights_orig(C_out * K_padded, 0);
    std::vector<uint8_t> Anew((C_out * K_padded) / 4, 0);

    std::mt19937 gen(42);
    std::uniform_int_distribution<> distBinary(0, 1);
    std::uniform_int_distribution<> distTernary(-1, 1);

    for (auto& x : image) x = distBinary(gen) ? 1 : -1;
    for (int n = 0; n < C_out; ++n) {
        for (int k = 0; k < K_actual; ++k) {
            weights_orig[n * K_padded + k] = distTernary(gen);
        }
    }

    encoder::encodeTern(weights_orig.data(), Anew.data(), C_out, K_padded);

    std::vector<int> res_naive(C_out * M, 0);
    std::vector<int> res_p_im2col(C_out * M, 0);

    conv2d_naive(image.data(), weights_orig.data(), res_naive.data(), 
                 C_in, C_out, H, W, K_h, K_w, stride, stride, pad, pad, K_padded);

    p_im2col_packed_conv(image.data(), Anew.data(), res_p_im2col.data(),
                         C_in, H, W, C_out, K_h, K_w, stride, pad, P, gemmV14_BLIS_SingleThread_Optimized);

    for (int i = 0; i < C_out * M; ++i) {
        ASSERT_EQ(res_naive[i], res_p_im2col[i]) 
            << "Ошибка в p-im2col на индексе " << i 
            << " (канал " << i / M << ", пиксель " << i % M << ")";
    }
}

TEST(Conv_Threads_Test, MultiThread_P_Im2Col_Correctness) {
    const int C_in = 8, H = 16, W = 16;
    const int C_out = 16, K_h = 3, K_w = 3;
    const int stride = 1, pad = 1;
    const int P = 32;
    const int n_threads = 4;

    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;
    int M = H_out * W_out;
    int K_actual = C_in * K_h * K_w;
    int K_padded = ((K_actual + 7) / 8) * 8;

    std::vector<int> image(C_in * H * W);
    std::vector<int> weights_orig(C_out * K_padded, 0);
    std::vector<uint8_t> Anew((C_out * K_padded) / 4, 0);
    std::vector<int> res_naive(C_out * M, 0);
    std::vector<int> res_threads(C_out * M, 0);

    std::mt19937 gen(1337);
    std::uniform_int_distribution<> distBinary(0, 1);
    std::uniform_int_distribution<> distTernary(-1, 1);

    for (auto& x : image) x = distBinary(gen) ? 1 : -1;
    for (int n = 0; n < C_out; ++n) {
        for (int k = 0; k < K_actual; ++k) {
            weights_orig[n * K_padded + k] = distTernary(gen);
        }
    }


    encoder::encodeTern(weights_orig.data(), Anew.data(), C_out, K_padded);

    conv2d_naive(image.data(), weights_orig.data(), res_naive.data(), 
                 C_in, C_out, H, W, K_h, K_w, stride, stride, pad, pad, K_padded);

    p_im2col_packed_conv_threads(image.data(), Anew.data(), res_threads.data(),
                                 C_in, H, W, C_out, K_h, K_w, stride, pad, P, n_threads,
                                gemmV14_BLIS_SingleThread_Optimized);

    for (int i = 0; i < C_out * M; ++i) {
        ASSERT_EQ(res_naive[i], res_threads[i]) 
            << "Mismatch at index " << i << " (Channel " << i/M << ")";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}