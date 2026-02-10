#include "util/encoder.h"

void gemmV0(int* A, int* B, int* C, int n, int m, int k);
void gemmV1(uint8_t* A0, uint8_t* A1, uint8_t* B, int* C, int n, int m, int k);
