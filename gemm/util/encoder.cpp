#include "encoder.h"

namespace encoder {
void baseEncodeTern(int* A, uint8_t* A0, uint8_t* A1, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A0[i * m + j] = A[i * m + j] == 1;
            A1[i * m + j] = A[i * m + j] == -1;
        }
    }
}

void baseEncodeBin(int* B, uint8_t* Bnew, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            Bnew[i * m + j] = B[i * m + j] == -1;
        }
    }
}

std::pair<uint8_t, uint8_t> mul(uint8_t a0, uint8_t a1, uint8_t b, uint8_t& c0, uint8_t& c1) {
    c0 = (a0 | b) & (a1 | !b);
    c1 = (a0 | !b) & (a1 | b);
}
};