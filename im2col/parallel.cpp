#include "conv.h"
#include <thread>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>

void p_im2col_worker(int m_start, int m_end, const int* image, uint8_t* Anew, int* output,
    int C_in, int H, int W, int C_out, int K_h, int K_w, int stride, int pad, int P, int M, int K_padded,
    gemm_func_t gemm_impl) {
    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;
    int K_actual = C_in * K_h * K_w;
    int K_bytes = K_padded / 8;

    std::vector<uint8_t> B_block(K_bytes * P);
    std::vector<int> res_block(C_out * P);

    for (int p_global_start = m_start; p_global_start < m_end; p_global_start += P) {
        int current_P = std::min(P, m_end - p_global_start);

        std::fill(B_block.begin(), B_block.begin() + K_bytes * current_P, 0);

        for (int p = 0; p < current_P; ++p) {
            int out_idx = p_global_start + p;
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

        gemm_impl(Anew, B_block.data(), res_block.data(), C_out, current_P, K_padded);

        for (int i = 0; i < C_out; ++i) {
            std::memcpy(
                &output[i * M + p_global_start], 
                &res_block[i * current_P], 
                current_P * sizeof(int)
            );
        }
    }
}

void p_im2col_packed_conv_threads(
    const int* image, uint8_t* Anew, int* output,
    int C_in, int H, int W, int C_out, int K_h, int K_w,
    int stride, int pad, int P, int num_threads,
    gemm_func_t gemm_impl
) {
    int H_out = (H + 2 * pad - K_h) / stride + 1;
    int W_out = (W + 2 * pad - K_w) / stride + 1;
    int M = H_out * W_out;
    int K_padded = ((C_in * K_h * K_w + 7) / 8) * 8;

    std::vector<std::thread> threads;
    int chunk_size = (M + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int m_start = t * chunk_size;
        int m_end = std::min(m_start + chunk_size, M);

        if (m_start < m_end) {
            threads.emplace_back(p_im2col_worker, 
                m_start, m_end, image, Anew, output,
                C_in, H, W, C_out, K_h, K_w, stride, pad, P, M, K_padded,
                gemm_impl
            );
        }
    }

    for (auto& th : threads) {
        th.join();
    }
}