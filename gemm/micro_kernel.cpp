#include "gemms.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>

#define MC 256
#define KC 256
#define NC 4096

inline __m256i fast_popcount_epi8_final(__m256i v) {
    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i low_mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
}

void gemmV12_AVX_4x32(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;
    __m256i mask_ff = _mm256_set1_epi8(0xFF);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) C[i * m + j] = 0;
    }

    for (int i = 0; i <= n - 4; i += 4) {
        for (int j = 0; j <= m - 32; j += 32) {
            __m256i acc[4][4];
            for(int r=0; r<4; ++r) { acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = _mm256_setzero_si256(); }

            for (int t = 0; t < k_bytes; t++) {
                __m256i v_b = _mm256_loadu_si256((__m256i*)&B[t * m + j]);
                __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);

                for(int r=0; r<4; ++r) {
                    __m256i va_p = _mm256_set1_epi8(A[((i+r) * k_bytes + t) * 2]);
                    __m256i va_n = _mm256_set1_epi8(A[((i+r) * k_bytes + t) * 2 + 1]);

                    __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b), _mm256_or_si256(va_n, v_not_b));
                    __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b), _mm256_or_si256(va_n, v_b));
                    __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));

                    __m128i diff_lo = _mm256_castsi256_si128(diff);
                    __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
                    acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
                    acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
                    acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
                    acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
                }
            }
            for(int r=0; r<4; ++r) {
                _mm256_storeu_si256((__m256i*)&C[(i+r) * m + j], acc[r][0]);
                _mm256_storeu_si256((__m256i*)&C[(i+r) * m + j + 8], acc[r][1]);
                _mm256_storeu_si256((__m256i*)&C[(i+r) * m + j + 16], acc[r][2]);
                _mm256_storeu_si256((__m256i*)&C[(i+r) * m + j + 24], acc[r][3]);
            }
        }
    }
    
    for (int i = (n / 4) * 4; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int t = 0; t < k_bytes; t++) {
                encoder::addMul(C[i * m + j], A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j]);
            }
        }
    }
    for (int i = 0; i < (n / 4) * 4; i++) {
        for (int j = (m / 32) * 32; j < m; j++) {
            for (int t = 0; t < k_bytes; t++) {
                encoder::addMul(C[i * m + j], A[(i * k_bytes + t) * 2], A[(i * k_bytes + t) * 2 + 1], B[t * m + j]);
            }
        }
    }
}

void pack_A_safe(int mc, int kc, const uint8_t* A, int k_bytes, uint8_t* A_packed) {
    for (int i = 0; i < mc; i += 4) {
        for (int k = 0; k < kc; ++k) {
            for (int r = 0; r < 4; ++r) {
                if (i + r < mc) {
                    A_packed[((i / 4) * kc + k) * 8 + r * 2]     = A[((i + r) * k_bytes + k) * 2];
                    A_packed[((i / 4) * kc + k) * 8 + r * 2 + 1] = A[((i + r) * k_bytes + k) * 2 + 1];
                } else {
                    A_packed[((i / 4) * kc + k) * 8 + r * 2]     = 0;
                    A_packed[((i / 4) * kc + k) * 8 + r * 2 + 1] = 0;
                }
            }
        }
    }
}

void pack_B_safe(int kc, int nc, const uint8_t* B, int m, uint8_t* B_packed) {
    for (int j = 0; j < nc; j += 32) {
        for (int k = 0; k < kc; ++k) {
            for (int c = 0; c < 32; ++c) {
                if (j + c < nc) {
                    B_packed[(j / 32 * kc + k) * 32 + c] = B[k * m + j + c];
                } else {
                    B_packed[(j / 32 * kc + k) * 32 + c] = 0;
                }
            }
        }
    }
}

void micro_kernel_final(int kc, const uint8_t* A_p, const uint8_t* B_p, int* C, int m, int cur_mc, int cur_nc) {
    __m256i acc[4][4];
    for(int i=0; i<4; ++i) { 
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = _mm256_setzero_si256(); 
    }

    __m256i mask_ff = _mm256_set1_epi8(0xFF);

    for (int k = 0; k < kc; ++k) {
        __m256i v_b = _mm256_loadu_si256((__m256i*)&B_p[k * 32]);
        __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);

        const uint8_t* a_ptr = &A_p[k * 8];

        for (int r = 0; r < 4; ++r) {
            __m256i va_p = _mm256_set1_epi8(a_ptr[r * 2]);
            __m256i va_n = _mm256_set1_epi8(a_ptr[r * 2 + 1]);
            
            __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b), _mm256_or_si256(va_n, v_not_b));
            __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b), _mm256_or_si256(va_n, v_b));
            
            __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));
            
            __m128i diff_lo = _mm256_castsi256_si128(diff);
            __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
            
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
            acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
            acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
        }
    }

    for (int r = 0; r < cur_mc; ++r) {
        int32_t temp[32];
        _mm256_storeu_si256((__m256i*)&temp[0], acc[r][0]);
        _mm256_storeu_si256((__m256i*)&temp[8], acc[r][1]);
        _mm256_storeu_si256((__m256i*)&temp[16], acc[r][2]);
        _mm256_storeu_si256((__m256i*)&temp[24], acc[r][3]);
        for (int c = 0; c < cur_nc; ++c) {
            C[r * m + c] += temp[c];
        }
    }
}

void gemmV14_BLIS_SingleThread(uint8_t* Anew, uint8_t* Bnew, int* resV1, int n, int m, int k) {
    int k_bytes = k / 8;
    memset(resV1, 0, n * m * sizeof(int));

    uint8_t* A_packed = (uint8_t*)_mm_malloc(MC * KC * 2, 64);
    uint8_t* B_packed = (uint8_t*)_mm_malloc(KC * NC, 64);

    for (int jc = 0; jc < m; jc += NC) {
        for (int ic = 0; ic < n; ic += MC) {
            int cur_nc = std::min(NC, m - jc);
            int cur_mc = std::min(MC, n - ic);

            for (int pc = 0; pc < k_bytes; pc += KC) {
                int cur_kc = std::min(KC, k_bytes - pc);

                pack_A_safe(cur_mc, cur_kc, &Anew[(ic * k_bytes + pc) * 2], k_bytes, A_packed);
                pack_B_safe(cur_kc, cur_nc, &Bnew[pc * m + jc], m, B_packed);

                for (int jr = 0; jr < cur_nc; jr += 32) {
                    for (int ir = 0; ir < cur_mc; ir += 4) {
                        int micro_nc = std::min(32, cur_nc - jr);
                        int micro_mc = std::min(4, cur_mc - ir);

                        micro_kernel_final(cur_kc, 
                                           &A_packed[(ir / 4 * cur_kc) * 8], 
                                           &B_packed[(jr / 32 * cur_kc) * 32], 
                                           &resV1[(ic + ir) * m + (jc + jr)], m, 
                                           micro_mc, micro_nc);
                    }
                }
            }
        }
    }
    _mm_free(A_packed);
    _mm_free(B_packed);
}

void gemmV16_BLIS_CorrectOrder(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k) {
    int k_bytes = k / 8;
    memset(res, 0, n * m * sizeof(int));

    uint8_t* A_packed = (uint8_t*)_mm_malloc(MC * KC * 2, 64);
    uint8_t* B_packed = (uint8_t*)_mm_malloc(KC * NC, 64);

    for (int jc = 0; jc < m; jc += NC) {
        int cur_nc = std::min(NC, m - jc);

        for (int pc = 0; pc < k_bytes; pc += KC) {
            int cur_kc = std::min(KC, k_bytes - pc);

            pack_B_safe(cur_kc, cur_nc, &Bnew[pc * m + jc], m, B_packed);

            for (int ic = 0; ic < n; ic += MC) {
                int cur_mc = std::min(MC, n - ic);

                pack_A_safe(cur_mc, cur_kc, &Anew[(ic * k_bytes + pc) * 2], k_bytes, A_packed);

                for (int jr = 0; jr < cur_nc; jr += 32) {
                    for (int ir = 0; ir < cur_mc; ir += 4) {
                        int micro_nc = std::min(32, cur_nc - jr);
                        int micro_mc = std::min(4, cur_mc - ir);

                        micro_kernel_final(cur_kc, 
                                           &A_packed[(ir / 4 * cur_kc) * 8], 
                                           &B_packed[(jr / 32 * cur_kc) * 32], 
                                           &res[(ic + ir) * m + (jc + jr)], m, 
                                           micro_mc, micro_nc);
                    }
                }
            }
        }
    }
    
    _mm_free(A_packed);
    _mm_free(B_packed);
}

void micro_kernel_v17(int kc, const uint8_t* A_p, const uint8_t* B_p, int* C, int m, int cur_mc, int cur_nc) {
    __m256i acc[4][4]; 
    for(int i=0; i<4; ++i) { 
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = _mm256_setzero_si256(); 
    }

    __m256i mask_ff = _mm256_set1_epi8(0xFF);

    int k = 0;
    for (; k <= kc - 2; k += 2) {
        __m256i v_b0 = _mm256_loadu_si256((__m256i*)&B_p[k * 32]);
        __m256i v_not_b0 = _mm256_andnot_si256(v_b0, mask_ff);
        const uint8_t* a_ptr0 = &A_p[k * 8];

        for (int r = 0; r < 4; ++r) {
            __m256i va_p = _mm256_set1_epi8(a_ptr0[r * 2]);
            __m256i va_n = _mm256_set1_epi8(a_ptr0[r * 2 + 1]);
            
            __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b0), _mm256_or_si256(va_n, v_not_b0));
            __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b0), _mm256_or_si256(va_n, v_b0));
            __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));
            
            __m128i diff_lo = _mm256_castsi256_si128(diff);
            __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
            acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
            acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
        }

        __m256i v_b1 = _mm256_loadu_si256((__m256i*)&B_p[(k + 1) * 32]);
        __m256i v_not_b1 = _mm256_andnot_si256(v_b1, mask_ff);
        const uint8_t* a_ptr1 = &A_p[(k + 1) * 8];

        for (int r = 0; r < 4; ++r) {
            __m256i va_p = _mm256_set1_epi8(a_ptr1[r * 2]);
            __m256i va_n = _mm256_set1_epi8(a_ptr1[r * 2 + 1]);
            
            __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b1), _mm256_or_si256(va_n, v_not_b1));
            __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b1), _mm256_or_si256(va_n, v_b1));
            __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));
            
            __m128i diff_lo = _mm256_castsi256_si128(diff);
            __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
            acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
            acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
        }
    }

    for (; k < kc; ++k) {
        __m256i v_b = _mm256_loadu_si256((__m256i*)&B_p[k * 32]);
        __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);
        const uint8_t* a_ptr = &A_p[k * 8];

        for (int r = 0; r < 4; ++r) {
            __m256i va_p = _mm256_set1_epi8(a_ptr[r * 2]);
            __m256i va_n = _mm256_set1_epi8(a_ptr[r * 2 + 1]);
            __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b), _mm256_or_si256(va_n, v_not_b));
            __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b), _mm256_or_si256(va_n, v_b));
            __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));
            __m128i diff_lo = _mm256_castsi256_si128(diff);
            __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
            acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
            acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
        }
    }

    if (cur_nc == 32 && cur_mc == 4) {
        for (int r = 0; r < 4; ++r) {
            __m256i c0 = _mm256_loadu_si256((__m256i*)&C[r * m + 0]);
            __m256i c1 = _mm256_loadu_si256((__m256i*)&C[r * m + 8]);
            __m256i c2 = _mm256_loadu_si256((__m256i*)&C[r * m + 16]);
            __m256i c3 = _mm256_loadu_si256((__m256i*)&C[r * m + 24]);

            c0 = _mm256_add_epi32(c0, acc[r][0]);
            c1 = _mm256_add_epi32(c1, acc[r][1]);
            c2 = _mm256_add_epi32(c2, acc[r][2]);
            c3 = _mm256_add_epi32(c3, acc[r][3]);

            _mm256_storeu_si256((__m256i*)&C[r * m + 0], c0);
            _mm256_storeu_si256((__m256i*)&C[r * m + 8], c1);
            _mm256_storeu_si256((__m256i*)&C[r * m + 16], c2);
            _mm256_storeu_si256((__m256i*)&C[r * m + 24], c3);
        }
    } else {
        for (int r = 0; r < cur_mc; ++r) {
            int32_t temp[32];
            _mm256_storeu_si256((__m256i*)&temp[0], acc[r][0]);
            _mm256_storeu_si256((__m256i*)&temp[8], acc[r][1]);
            _mm256_storeu_si256((__m256i*)&temp[16], acc[r][2]);
            _mm256_storeu_si256((__m256i*)&temp[24], acc[r][3]);
            for (int c = 0; c < cur_nc; ++c) {
                C[r * m + c] += temp[c];
            }
        }
    }
}

void gemmV17_BLIS_FastKernel(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k) {
    int k_bytes = k / 8;
    memset(res, 0, n * m * sizeof(int));
    uint8_t* A_packed = (uint8_t*)_mm_malloc(MC * KC * 2, 64);
    uint8_t* B_packed = (uint8_t*)_mm_malloc(KC * NC, 64);

    for (int jc = 0; jc < m; jc += NC) {
        int cur_nc = std::min(NC, m - jc);
        for (int pc = 0; pc < k_bytes; pc += KC) {
            int cur_kc = std::min(KC, k_bytes - pc);
            pack_B_safe(cur_kc, cur_nc, &Bnew[pc * m + jc], m, B_packed);
            for (int ic = 0; ic < n; ic += MC) {
                int cur_mc = std::min(MC, n - ic);
                pack_A_safe(cur_mc, cur_kc, &Anew[(ic * k_bytes + pc) * 2], k_bytes, A_packed);
                for (int jr = 0; jr < cur_nc; jr += 32) {
                    for (int ir = 0; ir < cur_mc; ir += 4) {
                        int micro_nc = std::min(32, cur_nc - jr);
                        int micro_mc = std::min(4, cur_mc - ir);
                        micro_kernel_v17(cur_kc, 
                                         &A_packed[(ir / 4 * cur_kc) * 8], 
                                         &B_packed[(jr / 32 * cur_kc) * 32], 
                                         &res[(ic + ir) * m + (jc + jr)], m, 
                                         micro_mc, micro_nc);
                    }
                }
            }
        }
    }
    _mm_free(A_packed);
    _mm_free(B_packed);
}

void micro_kernel_v18_prefetch(int kc, const uint8_t* A_p, const uint8_t* B_p, int* C, int m, int cur_mc, int cur_nc) {
    __m256i acc[4][4]; 
    for(int i=0; i<4; ++i) { 
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = _mm256_setzero_si256(); 
    }

    __m256i mask_ff = _mm256_set1_epi8(0xFF);

    for (int k = 0; k < kc; ++k) {
        __builtin_prefetch(&A_p[(k + 2) * 8], 0, 3);
        __builtin_prefetch(&B_p[(k + 2) * 32], 0, 3);

        __m256i v_b = _mm256_loadu_si256((__m256i*)&B_p[k * 32]);
        __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);
        const uint8_t* a_ptr = &A_p[k * 8];

        for (int r = 0; r < 4; ++r) {
            __m256i va_p = _mm256_set1_epi8(a_ptr[r * 2]);
            __m256i va_n = _mm256_set1_epi8(a_ptr[r * 2 + 1]);
            
            __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b), _mm256_or_si256(va_n, v_not_b));
            __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b), _mm256_or_si256(va_n, v_b));
            
            __m256i diff = _mm256_sub_epi8(fast_popcount_epi8_final(u_pos), fast_popcount_epi8_final(u_neg));
            
            __m128i diff_lo = _mm256_castsi256_si128(diff);
            __m128i diff_hi = _mm256_extracti128_si256(diff, 1);
            
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_cvtepi8_epi32(diff_lo));
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_lo, 8)));
            acc[r][2] = _mm256_add_epi32(acc[r][2], _mm256_cvtepi8_epi32(diff_hi));
            acc[r][3] = _mm256_add_epi32(acc[r][3], _mm256_cvtepi8_epi32(_mm_srli_si128(diff_hi, 8)));
        }
    }

    if (cur_nc == 32 && cur_mc == 4) {
        for (int r = 0; r < 4; ++r) {
            __m256i c0 = _mm256_loadu_si256((__m256i*)&C[r * m + 0]);
            __m256i c1 = _mm256_loadu_si256((__m256i*)&C[r * m + 8]);
            __m256i c2 = _mm256_loadu_si256((__m256i*)&C[r * m + 16]);
            __m256i c3 = _mm256_loadu_si256((__m256i*)&C[r * m + 24]);

            c0 = _mm256_add_epi32(c0, acc[r][0]);
            c1 = _mm256_add_epi32(c1, acc[r][1]);
            c2 = _mm256_add_epi32(c2, acc[r][2]);
            c3 = _mm256_add_epi32(c3, acc[r][3]);

            _mm256_storeu_si256((__m256i*)&C[r * m + 0], c0);
            _mm256_storeu_si256((__m256i*)&C[r * m + 8], c1);
            _mm256_storeu_si256((__m256i*)&C[r * m + 16], c2);
            _mm256_storeu_si256((__m256i*)&C[r * m + 24], c3);
        }
    } else {
        for (int r = 0; r < cur_mc; ++r) {
            int32_t temp[32];
            _mm256_storeu_si256((__m256i*)&temp[0], acc[r][0]);
            _mm256_storeu_si256((__m256i*)&temp[8], acc[r][1]);
            _mm256_storeu_si256((__m256i*)&temp[16], acc[r][2]);
            _mm256_storeu_si256((__m256i*)&temp[24], acc[r][3]);
            for (int c = 0; c < cur_nc; ++c) {
                C[r * m + c] += temp[c];
            }
        }
    }
}

void gemmV18_Ultimate_SingleThread(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k) {
    int k_bytes = k / 8;
    memset(res, 0, n * m * sizeof(int));
    uint8_t* A_packed = (uint8_t*)_mm_malloc(MC * KC * 2, 64);
    uint8_t* B_packed = (uint8_t*)_mm_malloc(KC * NC, 64);

    for (int jc = 0; jc < m; jc += NC) {
        int cur_nc = std::min(NC, m - jc);
        for (int pc = 0; pc < k_bytes; pc += KC) {
            int cur_kc = std::min(KC, k_bytes - pc);
            pack_B_safe(cur_kc, cur_nc, &Bnew[pc * m + jc], m, B_packed);
            for (int ic = 0; ic < n; ic += MC) {
                int cur_mc = std::min(MC, n - ic);
                pack_A_safe(cur_mc, cur_kc, &Anew[(ic * k_bytes + pc) * 2], k_bytes, A_packed);
                for (int jr = 0; jr < cur_nc; jr += 32) {
                    __builtin_prefetch(&res[ic * m + (jc + jr)], 1, 0);
                    
                    for (int ir = 0; ir < cur_mc; ir += 4) {
                        int micro_nc = std::min(32, cur_nc - jr);
                        int micro_mc = std::min(4, cur_mc - ir);
                        micro_kernel_v18_prefetch(cur_kc, 
                                         &A_packed[(ir / 4 * cur_kc) * 8], 
                                         &B_packed[(jr / 32 * cur_kc) * 32], 
                                         &res[(ic + ir) * m + (jc + jr)], m, 
                                         micro_mc, micro_nc);
                    }
                }
            }
        }
    }
    _mm_free(A_packed);
    _mm_free(B_packed);
}