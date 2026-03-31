#include "conv.h"

void conv2d_naive(const int* image, const int* weights, int* output, int C_in, int C_out, int H, int W,
    int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w, int K_padded, int pad_value) {

    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    int M = H_out * W_out;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                int sum = 0;
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    for (int kh = 0; kh < K_h; ++kh) {
                        for (int kw = 0; kw < K_w; ++kw) {
                            int h_in = h_out * stride_h - pad_h + kh;
                            int w_in = w_out * stride_w - pad_w + kw;
                            
                            int val = pad_value;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                val = image[c_in * H * W + h_in * W + w_in];
                            }
                            
                            int idx = c_in * K_h * K_w + kh * K_w + kw;
                            int w_idx = c_out * K_padded + idx;
                            
                            sum += val * weights[w_idx];
                        }
                    }
                }
                output[c_out * M + (h_out * W_out + w_out)] = sum;
            }
        }
    }
}