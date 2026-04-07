#include "module1.h"
#include <cstdint>
#include <cstddef>

/**
 * Versione a 64-bit del kernel di partizionamento.
 * In questa versione, part_id è un array di uint64_t.
 * Rappresenta il caso di massimo carico per il memory bus.
 */
void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint64_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP)
{
    // La maschera rimane coerente con il numero di partizioni P[cite: 18].
    uint64_t l_uMask = static_cast<uint64_t>(p_uP) - 1;

    // Informiamo il compilatore che non ci sono dipendenze tra le iterazioni.
    #pragma GCC ivdep
    for (size_t l_stI = 0; l_stI < p_stN; ++l_stI) {
        uint64_t l_uK = p_pKeys[l_stI];
        
        // Calcolo dell'hash deterministico tramite Shift-XOR[cite: 19].
        // Tutto il calcolo e la memorizzazione avvengono a 64 bit.
        p_pPart_id[l_stI] = (l_uK ^ (l_uK >> 32)) & l_uMask;
    }
}