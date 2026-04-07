#pragma once
#include <cstdint>
#include <cstddef>


inline void kernel_modulo(const uint64_t* __restrict__ p_pKeys, uint32_t* __restrict__ p_pPart_id, size_t p_stN, uint32_t p_uP, uint32_t p_uShift_val) {
    size_t l_stI;
    for (l_stI = 0; l_stI < p_stN; ++l_stI) {
        p_pPart_id[l_stI] = static_cast<uint32_t>(p_pKeys[l_stI] % p_uP);
    }
}


inline void kernel_bitmask(const uint64_t* __restrict__ p_pKeys, uint32_t* __restrict__ p_pPart_id, size_t p_stN, uint32_t p_uP, uint32_t p_uShift_val) {
    size_t l_stI;
    uint32_t l_uMask = p_uP - 1;
    for (l_stI = 0; l_stI < p_stN; ++l_stI) {
        p_pPart_id[l_stI] = static_cast<uint32_t>(p_pKeys[l_stI] & l_uMask);
    }
}

inline void kernel_shift_xor(const uint64_t* __restrict__ p_pKeys, uint32_t* __restrict__ p_pPart_id, size_t p_stN, uint32_t p_uP, uint32_t p_uShift_val) {
    size_t l_stI;
    uint32_t l_uMask = p_uP - 1;
    uint64_t l_uK;
    for (l_stI = 0; l_stI < p_stN; ++l_stI) {
        l_uK = p_pKeys[l_stI];
        p_pPart_id[l_stI] = static_cast<uint32_t>((l_uK ^ (l_uK >> 32)) & l_uMask);
    }
}

inline void kernel_fibonacci(const uint64_t* __restrict__ p_pKeys, uint32_t* __restrict__ p_pPart_id, size_t p_stN, uint32_t p_uP, uint32_t p_uShift_val) {
    size_t l_stI;
    const uint64_t l_uMagic = 11400714819323198485ULL;
    for (l_stI = 0; l_stI < p_stN; ++l_stI) {
        p_pPart_id[l_stI] = static_cast<uint32_t>((p_pKeys[l_stI] * l_uMagic) >> p_uShift_val);
    }
}

void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint64_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP);


void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint32_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP);
           
void compute_partitions(const uint64_t* __restrict__ p_pKeys, 
                        uint16_t* __restrict__ p_pPart_id, 
                        size_t p_stN, 
                        uint32_t p_uP);

void compute_partitions_cuda(const uint64_t* h_keys, 
                            uint16_t* h_part_id, 
                            size_t N, 
                            uint32_t P,
                            float& t_h2d,
                            float& t_kernel,
                            float& t_d2h);

