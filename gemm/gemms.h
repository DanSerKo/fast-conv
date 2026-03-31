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
void gemmV9_AVX_FastPopcnt(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV10_AVX_Unrolled(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV11_AVX_Broadcast(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV12_AVX_4x32(uint8_t* A, uint8_t* B, int* C, int n, int m, int k);
void gemmV14_BLIS_SingleThread(uint8_t* Anew, uint8_t* Bnew, int* resV1, int n, int m, int k);
void gemmV15_Ultimate(uint8_t* Anew, uint8_t* Bnew, int* resV1, int n, int m, int k);
void gemmV16_BLIS_CorrectOrder(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k);
void gemmV17_BLIS_FastKernel(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k);
void gemmV18_Ultimate_SingleThread(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k);
void gemmV14_BLIS_SingleThread_Optimized(uint8_t* Anew, uint8_t* Bnew, int* resV1, int n, int m, int k);
void gemmV19_TheUltimate_ST(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k);