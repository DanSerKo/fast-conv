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

void gemmV9_AVX_FastPopcnt(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    int k_blocks32 = k_bytes / 32; 

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            __m256i acc = _mm256_setzero_si256();
            
            int t = 0;
            for (; t < k_blocks32; t++) {
                __m256i b_val = _mm256_loadu_si256((__m256i*)&B[(t*32)*m + j]);
                __m256i a_pos = _mm256_loadu_si256((__m256i*)&A[(i * k_bytes + t*32) * 2]);
                __m256i a_neg = _mm256_loadu_si256((__m256i*)&A[(i * k_bytes + t*32) * 2 + 1]);
                
                __m256i not_b = _mm256_andnot_si256(b_val, _mm256_set1_epi8(0xFF));

                __m256i u_pos = _mm256_and_si256(_mm256_or_si256(a_pos, b_val), _mm256_or_si256(a_neg, not_b));
                __m256i u_neg = _mm256_and_si256(_mm256_or_si256(a_pos, not_b), _mm256_or_si256(a_neg, b_val));
                acc = _mm256_add_epi64(acc, popcount256_vec_sum(u_pos));
                acc = _mm256_sub_epi64(acc, popcount256_vec_sum(u_neg));
            }
            
            int sum = hsum_256(acc);
            
            for (int tail = t * 32; tail < k_bytes; tail++) {
                encoder::addMul(sum, A[(i * k_bytes + tail) * 2], A[(i * k_bytes + tail) * 2 + 1], B[tail * m + j]);
            }
            C[i * m + j] = sum;
        }
    }
}

void gemmV10_AVX_Unrolled(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    int k_blocks32 = k_bytes / 32; 

    for (int i = 0; i <= n - 2; i += 2) {
        for (int j = 0; j <= m - 2; j += 2) {
            
            __m256i acc00 = _mm256_setzero_si256();
            __m256i acc01 = _mm256_setzero_si256();
            __m256i acc10 = _mm256_setzero_si256();
            __m256i acc11 = _mm256_setzero_si256();

            int t = 0;
            for (; t < k_blocks32; t++) {
                __m256i a0_pos = _mm256_loadu_si256((__m256i*)&A[(i * k_bytes + t*32) * 2]);
                __m256i a0_neg = _mm256_loadu_si256((__m256i*)&A[(i * k_bytes + t*32) * 2 + 1]);
                
                __m256i a1_pos = _mm256_loadu_si256((__m256i*)&A[((i+1) * k_bytes + t*32) * 2]);
                __m256i a1_neg = _mm256_loadu_si256((__m256i*)&A[((i+1) * k_bytes + t*32) * 2 + 1]);

                __m256i b0 = _mm256_loadu_si256((__m256i*)&B[(t*32)*m + j]);
                __m256i b1 = _mm256_loadu_si256((__m256i*)&B[(t*32)*m + j + 1]);
                
                __m256i not_b0 = _mm256_andnot_si256(b0, _mm256_set1_epi8(0xFF));
                __m256i not_b1 = _mm256_andnot_si256(b1, _mm256_set1_epi8(0xFF));

                #define COMPUTE_CELL(ACC, APOS, ANEG, BVAL, BNOT) { \
                    __m256i u_pos = _mm256_and_si256(_mm256_or_si256(APOS, BVAL), _mm256_or_si256(ANEG, BNOT)); \
                    __m256i u_neg = _mm256_and_si256(_mm256_or_si256(APOS, BNOT), _mm256_or_si256(ANEG, BVAL)); \
                    ACC = _mm256_add_epi64(ACC, popcount256_vec_sum(u_pos)); \
                    ACC = _mm256_sub_epi64(ACC, popcount256_vec_sum(u_neg)); \
                }

                COMPUTE_CELL(acc00, a0_pos, a0_neg, b0, not_b0);
                COMPUTE_CELL(acc01, a0_pos, a0_neg, b1, not_b1);
                COMPUTE_CELL(acc10, a1_pos, a1_neg, b0, not_b0);
                COMPUTE_CELL(acc11, a1_pos, a1_neg, b1, not_b1);
                
                #undef COMPUTE_CELL
            }

            C[i*m + j] = hsum_256(acc00);
            C[i*m + j + 1] = hsum_256(acc01);
            C[(i+1)*m + j] = hsum_256(acc10);
            C[(i+1)*m + j + 1] = hsum_256(acc11);
        }
    }
}

void gemmV11_AVX_Broadcast(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    __m256i mask_ff = _mm256_set1_epi8(0xFF);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= m - 32; j += 32) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            for (int t = 0; t < k_bytes; t++) {
                uint8_t a_pos_val = A[(i * k_bytes + t) * 2];
                uint8_t a_neg_val = A[(i * k_bytes + t) * 2 + 1];
                __m256i v_a_pos = _mm256_set1_epi8(a_pos_val);
                __m256i v_a_neg = _mm256_set1_epi8(a_neg_val);

                __m256i v_b = _mm256_loadu_si256((__m256i*)&B[t * m + j]);
                __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);

                __m256i u_pos = _mm256_and_si256(_mm256_or_si256(v_a_pos, v_b), _mm256_or_si256(v_a_neg, v_not_b));
                __m256i u_neg = _mm256_and_si256(_mm256_or_si256(v_a_pos, v_not_b), _mm256_or_si256(v_a_neg, v_b));

                __m256i cnt_pos = popcount_per_byte(u_pos);
                __m256i cnt_neg = popcount_per_byte(u_neg);
                __m256i diff = _mm256_sub_epi8(cnt_pos, cnt_neg);

                __m128i d_lo = _mm256_castsi256_si128(diff);
                __m128i d_hi = _mm256_extracti128_si256(diff, 1);

                acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi8_epi32(d_lo));
                acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi8_epi32(_mm_srli_si128(d_lo, 8)));
                acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi8_epi32(d_hi));
                acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi8_epi32(_mm_srli_si128(d_hi, 8)));
            }

            _mm256_storeu_si256((__m256i*)&C[i * m + j], acc0);
            _mm256_storeu_si256((__m256i*)&C[i * m + j + 8], acc1);
            _mm256_storeu_si256((__m256i*)&C[i * m + j + 16], acc2);
            _mm256_storeu_si256((__m256i*)&C[i * m + j + 24], acc3);
        }

        for (int j_tail = (m / 32) * 32; j_tail < m; j_tail++) {
            int sum = 0;
            for (int t = 0; t < k_bytes; t++) {
                encoder::addMul(sum, A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j_tail]);
            }
            C[i * m + j_tail] = sum;
        }
    }
}