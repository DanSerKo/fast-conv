#include "gemms.h"

#include <algorithm>
#include <vector>

void gemmV5_blocked(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    const int BLOCK_N = 32;
    const int BLOCK_M = 32;
    const int BLOCK_K = 256;
    
    int k_bytes = k / 8;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = 0;
        }
    }

    for (int i0 = 0; i0 < n; i0 += BLOCK_N) {
        int i_max = std::min(i0 + BLOCK_N, n);
        for (int j0 = 0; j0 < m; j0 += BLOCK_M) {
            int j_max = std::min(j0 + BLOCK_M, m);
            for (int t0 = 0; t0 < k_bytes; t0 += (BLOCK_K / 8)) {
                int t_max = std::min(t0 + (BLOCK_K / 8), k_bytes);
                
                for (int i = i0; i < i_max; i++) {
                    for (int j = j0; j < j_max; j++) {
                        int sum = 0;
                        for (int t = t0; t < t_max; t++) {
                            encoder::addMul(sum, A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j]);
                        }
                        C[i * m + j] += sum;
                    }
                }
            }
        }
    }
}

void gemmV6_packed(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    const int BLOCK_N = 32;
    const int BLOCK_M = 32;
    const int BLOCK_K = 256; 
    int k_bytes = k / 8;
    
    std::vector<uint8_t> A_pack(BLOCK_N * (BLOCK_K / 8) * 2);
    std::vector<uint8_t> B_pack((BLOCK_K / 8) * BLOCK_M);

    for (int i = 0; i < n * m; i++) C[i] = 0;

    for (int i0 = 0; i0 < n; i0 += BLOCK_N) {
        int i_max = std::min(i0 + BLOCK_N, n);
        for (int t0 = 0; t0 < k_bytes; t0 += (BLOCK_K / 8)) {
            int t_max = std::min(t0 + (BLOCK_K / 8), k_bytes);
            
            packA(A, A_pack.data(), i0, i_max, t0, t_max, k_bytes);
            
            for (int j0 = 0; j0 < m; j0 += BLOCK_M) {
                int j_max = std::min(j0 + BLOCK_M, m);
                packB(B, B_pack.data(), j0, j_max, t0, t_max, m);
                
                for (int i = 0; i < (i_max - i0); i++) {
                    for (int j = 0; j < (j_max - j0); j++) {
                        int sum = 0;
                        for (int t = 0; t < (t_max - t0); t++) {
                            uint8_t a_pos = A_pack[(i * (t_max - t0) + t) * 2];
                            uint8_t a_neg = A_pack[(i * (t_max - t0) + t) * 2 + 1];
                            uint8_t b_val = B_pack[t * (j_max - j0) + j];
                            encoder::addMul(sum, a_pos, a_neg, b_val);
                        }
                        C[(i0 + i) * m + (j0 + j)] += sum;
                    }
                }
            }
        }
    }
}