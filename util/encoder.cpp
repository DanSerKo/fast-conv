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


void packA(uint8_t* A, uint8_t* A_pack, int i0, int i_max, int t0, int t_max, int k_bytes) {
    int idx = 0;
    for (int i = i0; i < i_max; i++) {
        for (int t = t0; t < t_max; t++) {
            A_pack[idx++] = A[(i * k_bytes + t) * 2];
            A_pack[idx++] = A[(i * k_bytes + t) * 2 + 1];
        }
    }
}

void packB(uint8_t* B, uint8_t* B_pack, int j0, int j_max, int t0, int t_max, int m) {
    int idx = 0;
    for (int t = t0; t < t_max; t++) {
        for (int j = j0; j < j_max; j++) {
            B_pack[idx++] = B[t * m + j];
        }
    }
}

int popcount128(__m128i v) {
    return __builtin_popcountll(_mm_extract_epi64(v, 0)) + 
           __builtin_popcountll(_mm_extract_epi64(v, 1));
}

int popcount256(__m256i v) {
    return __builtin_popcountll(_mm256_extract_epi64(v, 0)) + 
           __builtin_popcountll(_mm256_extract_epi64(v, 1)) +
           __builtin_popcountll(_mm256_extract_epi64(v, 2)) +
           __builtin_popcountll(_mm256_extract_epi64(v, 3));
}

__m256i popcount256_vec_sum(__m256i v) {
    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    
    __m256i pop_lo = _mm256_shuffle_epi8(lookup, lo);
    __m256i pop_hi = _mm256_shuffle_epi8(lookup, hi);
    __m256i pop = _mm256_add_epi8(pop_lo, pop_hi);
    
    return _mm256_sad_epu8(pop, _mm256_setzero_si256());
}

int hsum_256(__m256i v) {
    return _mm256_extract_epi64(v, 0) + 
           _mm256_extract_epi64(v, 1) + 
           _mm256_extract_epi64(v, 2) + 
           _mm256_extract_epi64(v, 3);
}

__m256i popcount_per_byte(__m256i v) {
    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    
    __m256i pop_lo = _mm256_shuffle_epi8(lookup, lo);
    __m256i pop_hi = _mm256_shuffle_epi8(lookup, hi);
    return _mm256_add_epi8(pop_lo, pop_hi);
}

