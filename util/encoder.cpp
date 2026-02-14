#include "encoder.h"

namespace encoder {
void encodeTern(int* A, uint8_t* Anew, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int ind = (i * m + j) / 8;
            int off = (i * m + j) % 8;
            Anew[2 * ind] |= (A[i * m + j] == 1) << off;
            Anew[2 * ind + 1] |= (A[i * m + j] == -1) << off;
        }
    }
}
void encodeBinT(int* B, uint8_t* Bnew, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            //int ind = (i * m + j) / 8;
            //int off = (i * m + j) % 8;
            int ind = (i / 8) * m + j;
            int off = i % 8;
            Bnew[ind] |= (B[i * m + j] == -1) << off;
        }
    }
}

void addMulSingle(int& c, uint8_t a0, uint8_t a1, uint8_t b) {
    c += (a0 | b) & (a1 | !b);
    c -= (a0 | !b) & (a1 | b);
}

void addMul(int& c, uint8_t a0, uint8_t a1, uint8_t b) {
    c += __builtin_popcount((a0 | b) & (a1 | ~b));
    c -= __builtin_popcount((a0 | ~b) & (a1 | b));
}
};
