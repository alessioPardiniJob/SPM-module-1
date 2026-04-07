#include "module1.h"
#include <cstdint>
#include <cstddef>


void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint64_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    // The mask remains consistent with the number of partitions P.
    uint64_t l_uMask = static_cast<uint64_t>(p_uP) - 1;

    // Hint to GCC that there are no loop-carried dependencies,
    // allowing it to safely auto-vectorize or unroll the loop if beneficial.
    #pragma GCC ivdep
    for (size_t l_stI = 0; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        
        // Deterministic hash calculation using a Shift-XOR approach.
        // The entire computation and subsequent storage are performed natively at 64 bits.
        p_pPart_id[l_stI] = (l_uK ^ (l_uK >> 32)) & l_uMask;
    }
}