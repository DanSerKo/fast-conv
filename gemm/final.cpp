#include "gemms.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>

#define MC 64
#define KC 256
#define NC 240

struct PopcountConstants {
    __m256i lookup;
    __m256i mask;
};

inline __m256i fast_popcount_reg(__m256i v, __m256i lookup, __m256i low_mask) {
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
}

void micro_kernel_4x24(int kc, const uint8_t* A_p, const uint8_t* B_p, int* C, int m, int cur_mc, int cur_nc) {
    __m256i acc00 = _mm256_setzero_si256(), acc01 = _mm256_setzero_si256(), acc02 = _mm256_setzero_si256();
    __m256i acc10 = _mm256_setzero_si256(), acc11 = _mm256_setzero_si256(), acc12 = _mm256_setzero_si256();
    __m256i acc20 = _mm256_setzero_si256(), acc21 = _mm256_setzero_si256(), acc22 = _mm256_setzero_si256();
    __m256i acc30 = _mm256_setzero_si256(), acc31 = _mm256_setzero_si256(), acc32 = _mm256_setzero_si256();

    const __m256i mask_ff = _mm256_set1_epi8(0xFF);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );

    int k = 0;
    for (; k <= kc - 4; k += 4) {
        for (int step = 0; step < 4; ++step) {
            int k_idx = k + step;
            __m256i v_b = _mm256_loadu_si256((__m256i*)&B_p[k_idx * 24]);
            __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);
            const uint8_t* a_ptr = &A_p[k_idx * 8];

            #define COMPUTE_ROW(row, acc0, acc1, acc2) { \
                __m256i va_p = _mm256_set1_epi8(a_ptr[row * 2]); \
                __m256i va_n = _mm256_set1_epi8(a_ptr[row * 2 + 1]); \
                __m256i u_pos = _mm256_and_si256(_mm256_or_si256(va_p, v_b), _mm256_or_si256(va_n, v_not_b)); \
                __m256i u_neg = _mm256_and_si256(_mm256_or_si256(va_p, v_not_b), _mm256_or_si256(va_n, v_b)); \
                __m256i diff = _mm256_sub_epi8(fast_popcount_reg(u_pos, lookup, low_mask), fast_popcount_reg(u_neg, lookup, low_mask)); \
                __m128i d_lo = _mm256_castsi256_si128(diff); \
                __m128i d_hi = _mm256_extracti128_si256(diff, 1); \
                acc0 = _mm256_add_epi32(acc0, _mm256_cvtepi8_epi32(d_lo)); \
                acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi8_epi32(_mm_srli_si128(d_lo, 8))); \
                acc2 = _mm256_add_epi32(acc2, _mm256_cvtepi8_epi32(d_hi)); \
            }

            COMPUTE_ROW(0, acc00, acc01, acc02);
            COMPUTE_ROW(1, acc10, acc11, acc12);
            COMPUTE_ROW(2, acc20, acc21, acc22);
            COMPUTE_ROW(3, acc30, acc31, acc32);
        }
    }
    for (; k < kc; ++k) {
        __m256i v_b = _mm256_loadu_si256((__m256i*)&B_p[k * 24]);
        __m256i v_not_b = _mm256_andnot_si256(v_b, mask_ff);
        const uint8_t* a_ptr = &A_p[k * 8];
        COMPUTE_ROW(0, acc00, acc01, acc02);
        COMPUTE_ROW(1, acc10, acc11, acc12);
        COMPUTE_ROW(2, acc20, acc21, acc22);
        COMPUTE_ROW(3, acc30, acc31, acc32);
    }

    auto store_row = [&](int r, __m256i a0, __m256i a1, __m256i a2) {
        if (r < cur_mc) {
            int* C_ptr = &C[r * m];
            if (cur_nc >= 24) {
                _mm256_storeu_si256((__m256i*)&C_ptr[0], _mm256_add_epi32(a0, _mm256_loadu_si256((__m256i*)&C_ptr[0])));
                _mm256_storeu_si256((__m256i*)&C_ptr[8], _mm256_add_epi32(a1, _mm256_loadu_si256((__m256i*)&C_ptr[8])));
                _mm256_storeu_si256((__m256i*)&C_ptr[16], _mm256_add_epi32(a2, _mm256_loadu_si256((__m256i*)&C_ptr[16])));
            } else {
                int32_t t[24];
                _mm256_storeu_si256((__m256i*)&t[0], a0);
                _mm256_storeu_si256((__m256i*)&t[8], a1);
                _mm256_storeu_si256((__m256i*)&t[16], a2);
                for(int c=0; c<cur_nc; ++c) C_ptr[c] += t[c];
            }
        }
    };

    store_row(0, acc00, acc01, acc02);
    store_row(1, acc10, acc11, acc12);
    store_row(2, acc20, acc21, acc22);
    store_row(3, acc30, acc31, acc32);
}

void pack_B_24(int kc, int nc, const uint8_t* B, int m, uint8_t* B_packed) {
    for (int j = 0; j < nc; j += 24) {
        int c_rem = std::min(24, nc - j);
        for (int k = 0; k < kc; ++k) {
            for (int c = 0; c < 24; ++c) {
                B_packed[(j / 24 * kc + k) * 24 + c] = (c < c_rem) ? B[k * m + j + c] : 0;
            }
        }
    }
}

void pack_A_safe_opt2(int mc, int kc, const uint8_t* A, int k_bytes, uint8_t* A_packed) {
    for (int i = 0; i < mc; i += 4) {
        int r_max = std::min(4, mc - i);
        if (r_max == 4) {
            for (int k = 0; k < kc; ++k) {
                for (int r = 0; r < 4; ++r) {
                    A_packed[((i / 4) * kc + k) * 8 + r * 2]     = A[((i + r) * k_bytes + k) * 2];
                    A_packed[((i / 4) * kc + k) * 8 + r * 2 + 1] = A[((i + r) * k_bytes + k) * 2 + 1];
                }
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                for (int r = 0; r < 4; ++r) {
                    if (r < r_max) {
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
}

void gemmV19_TheUltimate_ST(uint8_t* Anew, uint8_t* Bnew, int* res, int n, int m, int k) {
    int k_bytes = k / 8;
    memset(res, 0, n * m * sizeof(int));

    uint8_t* A_p = (uint8_t*)_mm_malloc(MC * KC * 2, 64);
    uint8_t* B_p = (uint8_t*)_mm_malloc(KC * NC, 64);

    for (int jc = 0; jc < m; jc += NC) {
        int cur_nc = std::min(NC, m - jc);
        for (int pc = 0; pc < k_bytes; pc += KC) {
            int cur_kc = std::min(KC, k_bytes - pc);
            pack_B_24(cur_kc, cur_nc, &Bnew[pc * m + jc], m, B_p);

            for (int ic = 0; ic < n; ic += MC) {
                int cur_mc = std::min(MC, n - ic);
                pack_A_safe_opt2(cur_mc, cur_kc, &Anew[(ic * k_bytes + pc) * 2], k_bytes, A_p);

                for (int jr = 0; jr < cur_nc; jr += 24) {
                    _mm_prefetch((const char*)&B_p[(jr / 24 + 1) * cur_kc * 24], _MM_HINT_T0);
                    
                    for (int ir = 0; ir < cur_mc; ir += 4) {
                        micro_kernel_4x24(cur_kc, 
                                          &A_p[(ir / 4 * cur_kc) * 8], 
                                          &B_p[(jr / 24 * cur_kc) * 24], 
                                          &res[(ic + ir) * m + (jc + jr)], m, 
                                          std::min(4, cur_mc - ir), std::min(24, cur_nc - jr));
                    }
                }
            }
        }
    }
    _mm_free(A_p); _mm_free(B_p);
}