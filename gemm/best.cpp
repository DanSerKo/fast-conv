#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>

#define MC 64
#define NC 256

// ------------------------------------------------------------
// Popcount for 256-bit vector of bytes (AVX2)
static inline __m256i popcnt_epi8(__m256i v) {
    const __m256i lut = _mm256_set_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    );
    __m256i lo = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
    __m256i cnt_lo = _mm256_shuffle_epi8(lut, lo);
    __m256i cnt_hi = _mm256_shuffle_epi8(lut, hi);
    return _mm256_add_epi8(cnt_lo, cnt_hi);
}

// ------------------------------------------------------------
// Pack 4 rows of A into a compact buffer (no change)
static void pack_A(int mc, int kc, const uint8_t* A, int k_bytes, uint8_t* Ap) {
    for (int i = 0; i < mc; i += 4) {
        int rmax = mc - i; if (rmax > 4) rmax = 4;
        for (int k = 0; k < kc; ++k) {
            for (int r = 0; r < 4; ++r) {
                if (r < rmax) {
                    Ap[((i/4)*kc + k)*8 + r*2]   = A[((i+r)*k_bytes + k)*2];
                    Ap[((i/4)*kc + k)*8 + r*2+1] = A[((i+r)*k_bytes + k)*2+1];
                } else {
                    Ap[((i/4)*kc + k)*8 + r*2] = Ap[((i/4)*kc + k)*8 + r*2+1] = 0;
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Pack B into blocks of 32 columns (AVX2 block size)
static void pack_B(int kc, int nc, const uint8_t* B, int m, uint8_t* Bp) {
    for (int j = 0; j < nc; j += 32) {
        int cmax = nc - j; if (cmax > 32) cmax = 32;
        uint8_t* dst = Bp + (j/32) * kc * 32;
        if (cmax == 32) {
            for (int k = 0; k < kc; ++k) {
                _mm256_storeu_si256((__m256i*)(dst + k*32),
                    _mm256_loadu_si256((const __m256i*)(B + k*m + j)));
            }
        } else {
            // generate byte mask: first cmax bytes = 0xFF, rest 0
            alignas(32) uint8_t mask_bytes[32];
            for (int i = 0; i < cmax; ++i) mask_bytes[i] = 0xFF;
            for (int i = cmax; i < 32; ++i) mask_bytes[i] = 0;
            __m256i mask = _mm256_loadu_si256((const __m256i*)mask_bytes);
            for (int k = 0; k < kc; ++k) {
                __m256i vec = _mm256_loadu_si256((const __m256i*)(B + k*m + j));
                vec = _mm256_and_si256(vec, mask);
                _mm256_storeu_si256((__m256i*)(dst + k*32), vec);
            }
        }
    }
}

// ------------------------------------------------------------
// Micro kernel: 4 rows × up to 32 columns, using AVX2
static inline void micro_kernel(
    int kc,
    const uint8_t* __restrict__ A_p,
    const uint8_t* __restrict__ B_p,
    int* __restrict__ C,
    int m,
    int cur_mc,
    int cur_nc
) {
    __m256i acc[4][2];   // each row: two 16‑int16 accumulators → total 32 int16
    for (int r = 0; r < 4; ++r) {
        acc[r][0] = _mm256_setzero_si256();
        acc[r][1] = _mm256_setzero_si256();
    }

    #pragma GCC unroll 7
    for (int k = 0; k < kc; ++k) {
        // load 32 bytes of B for this k
        __m256i vb = _mm256_loadu_si256((const __m256i*)(B_p + k*32));
        const uint8_t* a = A_p + k*8;

        for (int r = 0; r < 4; ++r) {
            __m256i vp = _mm256_set1_epi8((int8_t)a[r*2]);
            __m256i vn = _mm256_set1_epi8((int8_t)a[r*2+1]);

            __m256i not_vb = _mm256_andnot_si256(vb, _mm256_set1_epi8(-1));
            __m256i f1 = _mm256_or_si256(_mm256_and_si256(vp, vb),
                                         _mm256_and_si256(vn, not_vb));
            __m256i f2 = _mm256_or_si256(_mm256_and_si256(vp, not_vb),
                                         _mm256_and_si256(vn, vb));

            __m256i diff = _mm256_sub_epi8(popcnt_epi8(f1), popcnt_epi8(f2));

            // expand to 16‑bit: low 16 bytes -> first half, high 16 -> second half
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(diff));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(diff, 1));

            acc[r][0] = _mm256_add_epi16(acc[r][0], lo);
            acc[r][1] = _mm256_add_epi16(acc[r][1], hi);
        }
    }

    // Store results
    if (cur_nc == 32 && cur_mc == 4) {
        // full 4x32 block, store directly
        for (int r = 0; r < 4; ++r) {
            int* Cr = C + r*m;
            // first 16 columns from acc[r][0]
            __m256i lo16 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[r][0]));
            __m256i hi16 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[r][0], 1));
            _mm256_storeu_si256((__m256i*)(Cr + 0), lo16);
            _mm256_storeu_si256((__m256i*)(Cr + 8), hi16);
            // next 16 columns from acc[r][1]
            lo16 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[r][1]));
            hi16 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[r][1], 1));
            _mm256_storeu_si256((__m256i*)(Cr + 16), lo16);
            _mm256_storeu_si256((__m256i*)(Cr + 24), hi16);
        }
    } else {
        // partial block, use temporary array
        for (int r = 0; r < cur_mc; ++r) {
            int32_t tmp[32];
            _mm256_storeu_si256((__m256i*)&tmp[0],
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[r][0])));
            _mm256_storeu_si256((__m256i*)&tmp[8],
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[r][0], 1)));
            _mm256_storeu_si256((__m256i*)&tmp[16],
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc[r][1])));
            _mm256_storeu_si256((__m256i*)&tmp[24],
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc[r][1], 1)));
            for (int c = 0; c < cur_nc; ++c) C[r*m + c] = tmp[c];
        }
    }
}

// ------------------------------------------------------------
// Main GEMM routine (AVX2 version)
void gemmCandidate(uint8_t* A, uint8_t* B, int* C, int n, int m, int k) {
    int k_bytes = k / 8;                  // number of bytes per row/col
    // stack‑allocated aligned buffers
    alignas(64) uint8_t Ap[4096];        // MC/4 * kc * 8, kc ≤ 32 → 16*32*8 = 4096
    alignas(64) uint8_t Bp[8192];        // NC/32 * kc * 32 = 8*32*32 = 8192

    for (int jc = 0; jc < m; jc += NC) {
        int cnc = m - jc; if (cnc > NC) cnc = NC;
        pack_B(k_bytes, cnc, B + jc, m, Bp);

        for (int ic = 0; ic < n; ic += MC) {
            int cmc = n - ic; if (cmc > MC) cmc = MC;
            pack_A(cmc, k_bytes, A + (ic * k_bytes) * 2, k_bytes, Ap);

            for (int jr = 0; jr < cnc; jr += 32) {
                int mnc = cnc - jr; if (mnc > 32) mnc = 32;
                for (int ir = 0; ir < cmc; ir += 4) {
                    int mmc = cmc - ir; if (mmc > 4) mmc = 4;
                    micro_kernel(k_bytes,
                                 Ap + (ir/4) * k_bytes * 8,
                                 Bp + (jr/32) * k_bytes * 32,
                                 C + (ic + ir) * m + (jc + jr),
                                 m, mmc, mnc);
                }
            }
        }
    }
}