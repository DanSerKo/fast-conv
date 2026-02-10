#include "gemms.h"
#include <cassert>
#include <variant>

void gemmV1(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {  // "C" можно сделать variant
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[i * m + j] = 0;
        }
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < m; j++) {
                int indA = (i * k + t) / 4;
                int offA = ((i * k + t) % 4) * 2;

                int indB = (t * m + j) / 8;
                int offB = (t * m + j) % 8;

                encoder::addMul(C[i * m + j], (A[indA] >> offA) & 1, (A[indA] >> (offA + 1)) & 1, (B[indB] >> offB) & 1); 
            }
        }
    }
}
