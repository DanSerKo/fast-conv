#include "gemms.h"

#include <immintrin.h>

void gemmV1_avx512(__m512i* A0_simd, __m512i* A1_simd, uint8_t* B, int* C, int n, int m, int k) {
    const int bits_per_simd = 512;
    const int simd_per_row = (k + bits_per_simd - 1) / bits_per_simd;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j += 16) {
            _mm512_storeu_si512((__m512i*)(C + i * m + j), _mm512_setzero_si512());
        }
        
        for (int simd_idx = 0; simd_idx < simd_per_row; simd_idx++) {
            __m512i a0 = A0_simd[i * simd_per_row + simd_idx];
            __m512i a1 = A1_simd[i * simd_per_row + simd_idx];
            /*__mmask64 product_mask = _mm512_test_epi8_mask(a0, a1);
            
            if (product_mask == 0) {
                continue;
            }*/
            for (int bit = 0; bit < 512; bit++) {
                if ((product_mask >> bit) & 1) {
                    int t = simd_idx * 512 + bit;
                    if (t >= k) break;
                    
                    __m512i b_vec = _mm512_loadu_si512((__m512i*)(B + t * m));

                    for (int j = 0; j < m; j += 16) {
                        __m512i c_vec = _mm512_loadu_si512((__m512i*)(C + i * m + j));
                        c_vec = _mm512_add_epi32(c_vec, b_vec);
                        _mm512_storeu_si512((__m512i*)(C + i * m + j), c_vec);
                    }
                }
            }
        }
    }
}
