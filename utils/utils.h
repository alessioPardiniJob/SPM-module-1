#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#ifndef __CUDACC__
    #pragma GCC optimize("O3,no-tree-vectorize")
#endif

// Genera un array di N chiavi a 64 bit in modo deterministico partendo da un seed.
// key_space_size controlla il range dei valori e, indirettamente, la quantità di duplicati.
std::vector<uint64_t> generate_keys(size_t N, uint64_t seed, uint64_t key_space_size);

// Calcola un checksum semplice ma efficace sui risultati per verificare la correttezza del kernel.
uint64_t calculate_checksum(const uint16_t* part_id, size_t n);
