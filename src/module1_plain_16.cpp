#include <cstdint>
#include <cstddef>

// Nota: rimuovi l'unroll manuale dai commenti e da pragmas strani,
// lascia che GCC con -O3 decida quanto unrollare (di solito lo fa bene).
void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint16_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    // Funziona perfettamente finché P <= 65536 (il massimo per 16 bit)
    uint32_t l_uMask = p_uP - 1;

    // Suggeriamo a GCC che il ciclo è indipendente e ottimizzabile
    #pragma GCC ivdep
    for (size_t l_stI = 0; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        
        // Lo shift e lo XOR fatti rigorosamente a 64 bit,
        // maschera applicata, e infine si taglia a 16 bit con il cast.
        p_pPart_id[l_stI] = static_cast<uint16_t>((l_uK ^ (l_uK >> 32)) & l_uMask);
    }
}