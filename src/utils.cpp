#include "utils.h"
#include <random>
#include <array>
#include <iostream>  
#ifndef __CUDACC__
    #pragma GCC optimize("O3,no-tree-vectorize")
#endif


std::vector<uint64_t> generate_keys(size_t N, uint64_t seed, uint64_t key_space_size) {
    // 1. Prepariamo una sequenza di seed a 32-bit per inizializzare uniformemente 
    // lo stato interno del generatore.
    std::array<uint32_t, 2> seed_data = { 
        static_cast<uint32_t>(seed), 
        static_cast<uint32_t>(seed >> 32) 
    }; 
    std::seed_seq seq(seed_data.begin(), seed_data.end());
    
    // 2. Inizializzazione del generatore Mersenne Twister a 64 bit
    std::mt19937_64 generator(seq);
    
    // 3. Distribuzione uniforme nell'intervallo [0, key_space_size - 1]
    std::uniform_int_distribution<uint64_t> distribution(0, key_space_size - 1);
    
    // 4. Allocazione preventiva della memoria 
    std::vector<uint64_t> keys;
    keys.reserve(N);
    
    // 5. Popolamento dell'array
    for (size_t i = 0; i < N; ++i) {
        keys.push_back(distribution(generator));
    }
    
    return keys;
}

#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <cstddef>

/**
 * Generic calculation of position-dependent checksum.
 * T represents the input array width (uint16_t, uint32_t, or uint64_t).
 */
template <typename T>
uint64_t calculate_checksum(const T* part_id, size_t n) {
    uint64_t checksum = 0;
    for (size_t i = 0; i < n; ++i) {
        // Multiply by (i + 1) to ensure order-dependence and avoid zero-index nullification
        checksum += static_cast<uint64_t>(part_id[i]) * (i + 1);
        checksum ^= (checksum >> 32); 
    }
    return checksum;
}

/**
 * Generic reference mapping logic.
 * Computes the partition ID for a given key based on the number of partitions P.
 */
template <typename T>
inline T reference_mapping(uint64_t key, uint32_t P) {
    uint32_t mask = P - 1;
    return static_cast<T>((key ^ (key >> 32)) & mask);
}

/**
 * Validates the correctness of the generated partition IDs using Templates.
 * Works for any output type (uint16_t, uint32_t, uint64_t).
 */
template <typename T>
void verify_correctness(const uint64_t* keys, const T* part_id, size_t n, uint32_t P) {
    const size_t SMALL_N_THRESHOLD = 1000; 
    
    // 1. Calculate actual checksum from the optimized/SIMD results
    uint64_t actual_checksum = calculate_checksum(part_id, n);
    
    // 2. Calculate EXPECTED checksum using the exact SAME logic
    uint64_t expected_checksum = 0;
    for (size_t i = 0; i < n; ++i) {
        T ref_val = reference_mapping<T>(keys[i], P);
        
        // The logic here must perfectly match calculate_checksum()
        expected_checksum += static_cast<uint64_t>(ref_val) * (i + 1);
        expected_checksum ^= (expected_checksum >> 32);
    }

    // 3. Perform Detailed Verification based on input size
    if (n <= SMALL_N_THRESHOLD) {
        std::cout << "[*] N is small (" << n << " <= " << SMALL_N_THRESHOLD << ").\n";
        std::cout << "[*] Performing element-by-element comparison...\n";
        
        bool is_correct = true;
        for (size_t i = 0; i < n; ++i) {
            T expected = reference_mapping<T>(keys[i], P);
            if (part_id[i] != expected) {
                std::cerr << "[-] MISMATCH at index " << i 
                          << ": expected " << static_cast<uint64_t>(expected) 
                          << ", got " << static_cast<uint64_t>(part_id[i]) 
                          << " (Key: " << keys[i] << ")\n";
                is_correct = false;
                break; 
            }
        }

        // Display checksums for small N as well
        std::cout << "=> Got Checksum:      0x" << std::hex << actual_checksum << "\n";
        std::cout << "=> Expected Checksum: 0x" << expected_checksum << std::dec << "\n";
        
        if (is_correct) {
            std::cout << "[+] VERIFICATION PASSED: All elements match!\n";
        } else {
            std::cout << "[-] VERIFICATION FAILED: Individual element discrepancies found.\n";
        }

    } else {
        std::cout << "[*] N is large (" << n << " > " << SMALL_N_THRESHOLD << ").\n";
        std::cout << "[*] Performing Cross-Checksum Verification...\n";
        
        std::cout << "=> Got Checksum:      0x" << std::hex << actual_checksum << "\n";
        std::cout << "=> Expected Checksum: 0x" << expected_checksum << std::dec << "\n";

        if (actual_checksum == expected_checksum) {
            std::cout << "[+] VERIFICATION PASSED: Checksums match perfectly!\n";
        } else {
            std::cerr << "[-] VERIFICATION FAILED: Checksum mismatch!\n";
            std::cerr << "    Difference: " << (actual_checksum - expected_checksum) << "\n";
        }
    }
}



// =======================================================
// EXPLICIT TEMPLATE INSTANTIATIONS (Must be at the end of .cpp)
// =======================================================

template uint64_t calculate_checksum<uint16_t>(const uint16_t*, size_t);
template uint64_t calculate_checksum<uint32_t>(const uint32_t*, size_t);
template uint64_t calculate_checksum<uint64_t>(const uint64_t*, size_t);

template void verify_correctness<uint16_t>(const uint64_t*, const uint16_t*, size_t, uint32_t);
template void verify_correctness<uint32_t>(const uint64_t*, const uint32_t*, size_t, uint32_t);
template void verify_correctness<uint64_t>(const uint64_t*, const uint64_t*, size_t, uint32_t);