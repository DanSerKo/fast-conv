#include "../util/encoder.h"

void gemmV0(int* A, int* B, int* C, int n, int m, int k);
void gemmV1(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV2(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);