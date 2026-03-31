#include "conv.h"

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>

void p_im2col_packed_conv(const int* image, uint8_t* Anew, int* output,
    int C_in, int H, int W, int C_out, int K_h, int K_w,
    int stride, int pad, int P, gemm_func_t gemm_impl) {

    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;
    int M = H_out * W_out;
    int K_actual = C_in * K_h * K_w;
    int K_padded = ((K_actual + 7) / 8) * 8;
    int K_bytes = K_padded / 8;

    std::vector<uint8_t> B_block(K_bytes * P, 0);

    for (int p_start = 0; p_start < M; p_start += P) {
        int current_P = std::min(P, M - p_start);

        std::fill(B_block.begin(), B_block.end(), 0);

        for (int p = 0; p < current_P; ++p) {
            int out_idx = p_start + p;
            int h_out = out_idx / W_out;
            int w_out = out_idx % W_out;

            for (int i_row = 0; i_row < K_actual; ++i_row) {
                int c = i_row / (K_h * K_w);
                int kh = (i_row / K_w) % K_h;
                int kw = i_row % K_w;

                int h_in = h_out * stride - pad + kh;
                int w_in = w_out * stride - pad + kw;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    if (image[c * H * W + h_in * W + w_in] == -1) {
                        int byte_idx = (i_row / 8) * current_P + p;
                        B_block[byte_idx] |= (1 << (i_row % 8));
                    }
                }
            }
        }

        std::vector<int> res_block(C_out * current_P, 0);
        gemm_impl(Anew, B_block.data(), res_block.data(), C_out, current_P, K_padded);

        for (int i = 0; i < C_out; ++i) {
            std::memcpy(&output[i * M + p_start], &res_block[i * current_P], current_P * sizeof(int));
        }
    }
}