#include "util/encoder.h"

void gemmV1(uint8_t* A0, uint8_t* A1, uint8_t* B, int* C, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = 0;
        }
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < m; j++) {
                encoder::addMul(C[i * m + j], A0[i * k + t], A1[i * k + t], B[t * m + j]); 
            }
        }
    }
}
