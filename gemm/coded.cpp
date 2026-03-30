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
                int indA = (i * k + t) / 8;
                int offA = (i * k + t) % 8;

                int indB = (t / 8) * m + j;
                int offB = t % 8;

                encoder::addMulSingle(C[i * m + j], (A[indA * 2] >> offA) & 1, (A[indA * 2 + 1] >> offA) & 1, (B[indB] >> offB) & 1); 
            }
        }
    }
}

void gemmV2(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_block = k / 8;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int sum = 0;
            for (int t = 0; t < k_block; t++) {
                encoder::addMul(sum, A[(i * k_block + t) * 2], A[(i * k_block + t) * 2 + 1], B[t * m + j]);
            }
            C[i * m + j] = sum;
        }
    }
}

void gemmV3(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_block = k / 8;
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < k_block; t++) {
            for (int j = 0; j < m; j++) {
                encoder::addMul(C[i * m + j], A[(i * k_block + t) * 2], A[(i * k_block + t) * 2 + 1], B[t * m + j]);
            }
        }
    }
}

void gemmV4(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_block = k / 8;
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int t = 0; t < k_block; t++) {
                encoder::addMul(sum, A[(i * k_block + t) * 2], A[(i * k_block + t) * 2 + 1], B[t * m + j]);
            }
            C[i * m + j] = sum;
        }
    }
}

