// Strictly control the test file: NO auto-vectorization here.
// We want to measure only the kernel performance, not the surrounding boilerplate code.
#pragma GCC optimize("O3,no-tree-vectorize")

#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <string>

#include "hpc_helpers.hpp"
#include "utils.h"
#include "module1.h"

using namespace std;

int main(int argc, char** argv) {
    // ==========================================
    // 1. VARIABLE DECLARATION
    // ==========================================
    size_t              l_stN;
    uint32_t            l_uP;
    uint64_t            l_uSeed;
    int                 l_iRUNS;
    vector<double>      l_vdTimes_ms;
    int                 l_i;
    
    double              l_dMedian_ms;
    double              l_dSum;
    double              l_dMean;
    double              l_dSq_sum;
    double              l_dStd_dev;
    double              l_dThroughput;
    
    // Set target type to 32-bit
    typedef uint32_t    target_t;
    volatile target_t   l_uDummy;
    
    // Vectors
    vector<uint64_t>            l_vKeys;
    vector<no_init_t<target_t>> l_vPart_id;

    // ==========================================
    // 2. SETUP AND ALLOCATION
    // ==========================================
    l_stN = 50'000'000; 
    l_uP = 4096;
    l_uSeed = 42;
    l_iRUNS = 10;
    
    l_vdTimes_ms.resize(l_iRUNS); // Allocate space for timing results

    cout << "========================================================\n";
    cout << "           BENCHMARK EXECUTION (N = 50M - 32bit)       \n";
    cout << "========================================================\n";

    cout << "[*] Generating keys and allocating memory...\n";
    // ⚠️ CRITICAL: Generate and resize BEFORE performing warm-up
    l_vKeys = generate_keys(l_stN, l_uSeed, UINT64_MAX); 
    l_vPart_id.resize(l_stN);

    // Check alignment for AVX2/AVX-512 safety
    if ((reinterpret_cast<uintptr_t>(l_vKeys.data()) % 32 != 0) || 
        (reinterpret_cast<uintptr_t>(l_vPart_id.data()) % 32 != 0)) {
        cout << "[!] WARNING: Memory not aligned to 32 bytes!\n";
    }

    // ==========================================
    // 3. WARM-UP & FIRST-TOUCH POLICY
    // ==========================================
    // Now that the array exists, force the O.S. to physically allocate 
    // the memory pages by zeroing out the entire output array.
    std::fill(l_vPart_id.begin(), l_vPart_id.end(), 0);
    
    // Small warm-up for instruction caches
    compute_partitions(l_vKeys.data(), (target_t*)l_vPart_id.data(), 1000, l_uP);


    // ==========================================
    // 4. BENCHMARK LOOP (10 RUNS)
    // ==========================================
    cout << "[*] Executing Kernel (" << l_iRUNS << " runs)...\n";
    
    for (l_i = 0; l_i < l_iRUNS; ++l_i) {
        
        TIMERSTART(Kernel)
        
        // UNIVERSAL KERNEL CALL
        compute_partitions(l_vKeys.data(), (target_t*)l_vPart_id.data(), l_stN, l_uP);

        TIMERSTOP(Kernel)
        
        l_vdTimes_ms[l_i] = elapsed_Kernel * 1000.0;
    }

    // ==========================================
    // 5. STATISTICS CALCULATION
    // ==========================================
    sort(l_vdTimes_ms.begin(), l_vdTimes_ms.end());
    l_dMedian_ms = (l_vdTimes_ms[l_iRUNS/2 - 1] + l_vdTimes_ms[l_iRUNS/2]) / 2.0;

    l_dSum = accumulate(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), 0.0);
    l_dMean = l_dSum / l_iRUNS;
    l_dSq_sum = inner_product(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), l_vdTimes_ms.begin(), 0.0);
    l_dStd_dev = sqrt(l_dSq_sum / l_iRUNS - l_dMean * l_dMean);

    l_dThroughput = (l_stN / (l_dMedian_ms / 1000.0)) / 1000000.0;

    // ==========================================
    // 6. PRINT RESULTS
    // ==========================================
    cout << "\n=> STATISTICAL RESULTS:\n";
    cout << "   - Median  : " << fixed << setprecision(2) << l_dMedian_ms << " ms\n";
    cout << "   - Std Dev : " << l_dStd_dev << " ms\n";
    cout << "   - T-put   : " << l_dThroughput << " Mkeys/s\n";
    cout << "========================================================\n";

    // ANTI-OPTIMIZATION TRICK
    l_uDummy = ((target_t*)l_vPart_id.data())[l_stN / 2]; 
    (void)l_uDummy;

    // ==========================================
    // 7. CORRECTNESS VERIFICATION
    // ==========================================
    // Uses the template function: verify_correctness<uint32_t>
    verify_correctness(l_vKeys.data(), (target_t*)l_vPart_id.data(), l_stN, l_uP);
    
    cout << "========================================================\n";

    return 0;
}