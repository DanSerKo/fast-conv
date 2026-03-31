#include "../gemm/gemms.h"

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>

typedef void (*gemm_func_t)(uint8_t*, uint8_t*, int*, int, int, int);

void im2col_packed_bin(
    const int* image, uint8_t* Bnew, int C_in, int H, int W,
    int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int pad_value = 1);

void conv2d_naive(
    const int* image, const int* weights, int* output,
    int C_in, int C_out, int H, int W,
    int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int K_padded, int pad_value = 1);

void p_im2col_packed_conv(
    const int* image, uint8_t* Anew, int* output,
    int C_in, int H, int W, int C_out, int K_h, int K_w,
    int stride, int pad, int P, gemm_func_t gemm_impl);

void p_im2col_packed_conv_threads(
    const int* image, uint8_t* Anew, int* output,
    int C_in, int H, int W, int C_out, int K_h, int K_w,
    int stride, int pad, int P, int num_threads,
    gemm_func_t gemm_impl);