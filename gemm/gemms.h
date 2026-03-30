#include "../util/encoder.h"

void gemmV0(int* A, int* B, int* C, int n, int m, int k);
void gemmV1(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV2(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV3(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV4(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV5_blocked(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV6_packed(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV7_SSE(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV8_AVX(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);