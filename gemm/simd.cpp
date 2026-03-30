#include "gemms.h"

#include <immintrin.h>

#include <immintrin.h>


void gemmV7_SSE(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int sum = 0;
            int t = 0;
            for (; t <= k_bytes - 16; t += 16) {
                __m128i a0 = _mm_loadu_si128((__m128i*)&A[(i * k_bytes + t) * 2]);
                uint8_t buf_a0[16], buf_a1[16], buf_b[16];
                for(int s=0; s<16; s++) {
                    buf_a0[s] = A[(i * k_bytes + t + s) * 2];
                    buf_a1[s] = A[(i * k_bytes + t + s) * 2 + 1];
                    buf_b[s]  = B[(t + s) * m + j];
                }
                
                __m128i va0 = _mm_loadu_si128((__m128i*)buf_a0);
                __m128i va1 = _mm_loadu_si128((__m128i*)buf_a1);
                __m128i vb  = _mm_loadu_si128((__m128i*)buf_b);
                __m128i vnot_b = _mm_andnot_si128(vb, _mm_set1_epi8(0xFF));
                __m128i pos_part = _mm_and_si128(_mm_or_si128(va0, vb), _mm_or_si128(va1, vnot_b));
                __m128i neg_part = _mm_and_si128(_mm_or_si128(va0, vnot_b), _mm_or_si128(va1, vb));
                
                sum += popcount128(pos_part) - popcount128(neg_part);
            }
            for (; t < k_bytes; t++) {
                encoder::addMul(sum, A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j]);
            }
            C[i * m + j] = sum;
        }
    }
}


void gemmV8_AVX(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int sum = 0;
            int t = 0;
            for (; t <= k_bytes - 32; t += 32) {
                uint8_t buf_a0[32], buf_a1[32], buf_b[32];
                for(int s=0; s<32; s++) {
                    buf_a0[s] = A[(i * k_bytes + t + s) * 2];
                    buf_a1[s] = A[(i * k_bytes + t + s) * 2 + 1];
                    buf_b[s]  = B[(t + s) * m + j];
                }

                __m256i va0 = _mm256_loadu_si256((__m256i*)buf_a0);
                __m256i va1 = _mm256_loadu_si256((__m256i*)buf_a1);
                __m256i vb  = _mm256_loadu_si256((__m256i*)buf_b);
                __m256i vnot_b = _mm256_andnot_si256(vb, _mm256_set1_epi8(0xFF));

                __m256i v_pos = _mm256_and_si256(_mm256_or_si256(va0, vb), _mm256_or_si256(va1, vnot_b));
                __m256i v_neg = _mm256_and_si256(_mm256_or_si256(va0, vnot_b), _mm256_or_si256(va1, vb));

                sum += popcount256(v_pos) - popcount256(v_neg);
            }
            for (; t < k_bytes; t++) {
                encoder::addMul(sum, A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j]);
            }
            C[i * m + j] = sum;
        }
    }
}