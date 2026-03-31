#include <cstdint>
#include <utility>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

namespace encoder {
void encodeTern(int* A, uint8_t* Anew, int n, int m);
void encodeBinT(int* B, uint8_t* Bnew, int n, int m);

void addMulSingle(int& c, uint8_t a0, uint8_t a1, uint8_t b);
void addMul(int& c, uint8_t a0, uint8_t a1, uint8_t b);
};

void packA(uint8_t* A, uint8_t* A_pack, int i0, int i_max, int t0, int t_max, int k_bytes);
void packB(uint8_t* B, uint8_t* B_pack, int j0, int j_max, int t0, int t_max, int m);
int popcount128(__m128i v);
int popcount256(__m256i v);

int hsum_256(__m256i v);
__m256i popcount256_vec_sum(__m256i v); 
__m256i popcount_per_byte(__m256i v);