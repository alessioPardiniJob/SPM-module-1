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


#include <cstdint>
#include <cstddef>

uint64_t calculate_checksum(const uint16_t* part_id, size_t n) {
    uint64_t checksum = 0;
    for (size_t i = 0; i < n; ++i) {
        // Moltiplichiamo per (i + 1) per evitare che uno zero all'indice 0 annulli il calcolo.
        // L'operazione rende il checksum dipendente dalla posizione esatta dell'elemento.
        checksum += static_cast<uint64_t>(part_id[i]) * (i + 1);
        
        checksum ^= (checksum >> 32); 
    }
    return checksum;
}




uint16_t reference_mapping(uint64_t key, uint32_t P) {

    uint32_t mask = P - 1;
    
    return static_cast<uint16_t>((key ^ (key >> 32)) & mask);
}

// La funzione "intelligente" che sceglie cosa fare in base a N
void verify_correctness(const uint64_t* keys, const uint16_t* part_id, size_t n, uint32_t P) {
    // Definiamo una soglia per l'input "piccolo"
    const size_t SMALL_N_THRESHOLD = 1000; 

    if (n <= SMALL_N_THRESHOLD) {
        std::cout << "[*] N e' piccolo (" << n << " <= " << SMALL_N_THRESHOLD << ").\n";
        std::cout << "[*] Eseguo element-by-element comparison (Reference vs Ottimizzata)...\n";
        
        bool is_correct = true;
        for (size_t i = 0; i < n; ++i) {
            uint16_t expected = reference_mapping(keys[i], P);
            if (part_id[i] != expected) {
                std::cerr << "[-] MISMATCH all'indice " << i 
                          << ": expected " << expected 
                          << ", got " << part_id[i] << " (Key: " << keys[i] << ")\n";
                is_correct = false;
                break; // Si ferma al primo errore, rimuovi se vuoi vedere tutto
            }
        }
        
        if (is_correct) {
            std::cout << "[+] VERIFICA PASSATA: Tutti gli elementi coincidono!\n";
        } else {
            std::cout << "[-] VERIFICA FALLITA: Trovate discrepanze.\n";
        }
    } else {
        std::cout << "[*] N e' grande (" << n << " > " << SMALL_N_THRESHOLD << ").\n";
        std::cout << "[*] Calcolo checksum per verifica (element-by-element saltato)...\n";
        
        uint64_t final_checksum = calculate_checksum(part_id, n);
        std::cout << "=> CHECKSUM: 0x" << std::hex << final_checksum << std::dec << "\n";
    }
}