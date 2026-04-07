#include "module1.h"
#include <immintrin.h>
#include <cstdint>

void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint16_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    uint32_t l_uMask = p_uP - 1;
    size_t l_stI = 0;

    // 1. PROLOGO
    while (l_stI < p_stN && (reinterpret_cast<uintptr_t>(&p_pPart_id[l_stI]) % 32) != 0) {
        uint64_t l_uK = p_pKeys[l_stI];
        p_pPart_id[l_stI] = static_cast<uint16_t>((l_uK ^ (l_uK >> 32))) & l_uMask;
        ++l_stI;
    }

    // 2. CICLO VETTORIZZATO
    __m256i v_mask = _mm256_set1_epi32(l_uMask);
    size_t l_stLimit = l_stI + ((p_stN - l_stI) & ~15ULL); 

    for (; l_stI < l_stLimit; l_stI += 16) {
        
        // IL TOCCO FINALE: Prefetch software. 
        // 128 byte avanti = 16 chiavi (la prossima iterazione)
        _mm_prefetch((const char*)&p_pKeys[l_stI + 64], _MM_HINT_NTA);

        __m256i k1 = _mm256_loadu_si256((const __m256i*)&p_pKeys[l_stI + 0]);
        __m256i k2 = _mm256_loadu_si256((const __m256i*)&p_pKeys[l_stI + 4]);
        __m256i k3 = _mm256_loadu_si256((const __m256i*)&p_pKeys[l_stI + 8]);
        __m256i k4 = _mm256_loadu_si256((const __m256i*)&p_pKeys[l_stI + 12]);

        __m256i x1 = _mm256_xor_si256(k1, _mm256_srli_epi64(k1, 32));
        __m256i x2 = _mm256_xor_si256(k2, _mm256_srli_epi64(k2, 32));
        __m256i x3 = _mm256_xor_si256(k3, _mm256_srli_epi64(k3, 32));
        __m256i x4 = _mm256_xor_si256(k4, _mm256_srli_epi64(k4, 32));

        __m256i p1_mixed = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x1), _mm256_castsi256_ps(x2), _MM_SHUFFLE(2, 0, 2, 0)));
        __m256i p2_mixed = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x3), _mm256_castsi256_ps(x4), _MM_SHUFFLE(2, 0, 2, 0)));

        __m256i p1 = _mm256_and_si256(_mm256_permute4x64_epi64(p1_mixed, 0xD8), v_mask);
        __m256i p2 = _mm256_and_si256(_mm256_permute4x64_epi64(p2_mixed, 0xD8), v_mask);

        __m256i packed_16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(p1, p2), 0xD8);

        _mm256_stream_si256((__m256i*)&p_pPart_id[l_stI], packed_16);
    }

    // 3. EPILOGO
    for (; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        p_pPart_id[l_stI] = static_cast<uint16_t>((l_uK ^ (l_uK >> 32))) & l_uMask;
    }
}