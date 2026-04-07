#include "module1.h"


void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint32_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    uint32_t l_uMask = p_uP - 1;

    // Suggest to GCC that the loop is independent and optimizable
    #pragma GCC ivdep
    for (size_t l_stI = 0; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        
        // The shift and XOR strictly performed at 64 bits, 
        // and finally truncated to 32 bits with the cast.
        p_pPart_id[l_stI] = static_cast<uint32_t>((l_uK ^ (l_uK >> 32))) & l_uMask;
    }
}