#include "module1.h"
#include <immintrin.h>

// ====================================================================
// STEP 1: Naive AVX2 Vectorization
// Literal translation of the approach "Load -> Hash -> Extract -> Store"
// ====================================================================
void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint16_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    uint32_t l_uMask = p_uP - 1;
    size_t l_stI = 0;

    // Compute the limit to process full blocks of 4 keys (256 bits)
    size_t l_stLimit = p_stN & ~3ULL; 

    for (; l_stI < l_stLimit; l_stI += 4) {
        
        // 1. COMPUTATIONAL PHASE (Efficient)
        // Load 4 64-bit keys into the YMM register
        __m256i keys = _mm256_loadu_si256((const __m256i*)&p_pKeys[l_stI]);
        
        // Apply hashing logic (Shift + XOR) in parallel
        __m256i shifted = _mm256_srli_epi64(keys, 32);
        __m256i hashed  = _mm256_xor_si256(keys, shifted);

        // 2. NAIVE EXTRACTION APPROACH (The Bottleneck)
        // Extract individually the 4 64-bit elements to scalar registers (GPRs).
        // This causes the "Domain Crossing Penalty".
        uint64_t h0 = _mm256_extract_epi64(hashed, 0);
        uint64_t h1 = _mm256_extract_epi64(hashed, 1);
        uint64_t h2 = _mm256_extract_epi64(hashed, 2);
        uint64_t h3 = _mm256_extract_epi64(hashed, 3);

        // 3. SCALAR STORAGE
        // Cast to 16-bit, apply the mask, and write to memory serially.
        p_pPart_id[l_stI + 0] = static_cast<uint16_t>(h0) & l_uMask;
        p_pPart_id[l_stI + 1] = static_cast<uint16_t>(h1) & l_uMask;
        p_pPart_id[l_stI + 2] = static_cast<uint16_t>(h2) & l_uMask;
        p_pPart_id[l_stI + 3] = static_cast<uint16_t>(h3) & l_uMask;
    }

    // ====================================================================
    // SCALAR EPILOGUE
    // Process the remaining elements if N is not a multiple of 4
    // ====================================================================
    for (; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        p_pPart_id[l_stI] = static_cast<uint16_t>((l_uK ^ (l_uK >> 32))) & l_uMask;
    }
}