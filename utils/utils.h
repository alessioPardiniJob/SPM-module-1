#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

#ifndef __CUDACC__
    #pragma GCC optimize("O3,no-tree-vectorize")
#endif

// Generates an array of N 64-bit keys deterministically from a seed.
// key_space_size controls the value range and, indirectly, the amount of duplicates.
std::vector<uint64_t> generate_keys(size_t N, uint64_t seed, uint64_t key_space_size);

// Calculates a simple but effective checksum on the results to verify kernel correctness.
// Templated to support uint16_t, uint32_t, and uint64_t output arrays.
template <typename T>
uint64_t calculate_checksum(const T* part_id, size_t n);

// Validates the correctness of the generated partition IDs.
template <typename T>
void verify_correctness(const uint64_t* keys, const T* part_id, size_t n, uint32_t P);