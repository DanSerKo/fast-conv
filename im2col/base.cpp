#include "conv.h"

#include <cstdint>
#include <cstring>
#include <vector>

void im2col_packed_bin(const int* image, uint8_t* Bnew, int C_in, int H, int W,
    int K_h, int K_w, int stride_h, int stride_w, int pad_h, int pad_w, int pad_value) {

    int H_out = (H + 2 * pad_h - K_h) / stride_h + 1;
    int W_out = (W + 2 * pad_w - K_w) / stride_w + 1;
    int M = H_out * W_out;
    
    int K = C_in * K_h * K_w;
    int K_padded = ((K + 7) / 8) * 8;
    int K_bytes = K_padded / 8;
    
    memset(Bnew, 0, K_bytes * M * sizeof(uint8_t));
    
    for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
            int col_idx = h_out * W_out + w_out;
            
            for (int c = 0; c < C_in; ++c) {
                for (int kh = 0; kh < K_h; ++kh) {
                    for (int kw = 0; kw < K_w; ++kw) {
                        int row_idx = c * K_h * K_w + kh * K_w + kw;
                        
                        int h_in = h_out * stride_h - pad_h + kh;
                        int w_in = w_out * stride_w - pad_w + kw;
                        
                        int val = pad_value;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            val = image[c * H * W + h_in * W + w_in];
                        }
                        
                        int byte_idx = (row_idx / 8) * M + col_idx;
                        int bit_off = row_idx % 8;
                        
                        if (val == -1) {
                            Bnew[byte_idx] |= (1 << bit_off);
                        }
                    }
                }
            }
        }
    }
}