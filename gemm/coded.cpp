#include "util/encoder.h"

void gemmV1(uint8_t* A0, uint8_t* A1, uint8_t* B, uint8_t* C0, uint8_t* C1, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < m; j++) {
                encoder::mul(A0[i * k + t], A1[i * k + t], B[t * m + j], C0[i * m + j], C1[i * m + j]); 
            }
        }
    }
}
