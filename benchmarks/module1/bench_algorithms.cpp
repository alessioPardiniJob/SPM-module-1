// Force the compiler to use O3 but disable loop vectorization.
// This guarantees that the scalar baseline is pure and not automatically optimized via SIMD.
#pragma GCC optimize("O3,no-tree-vectorize")

#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

// Include the professor's HPC header for no_init_t, TIMERSTART, and TIMERSTOP
#include "hpc_helpers.hpp"
#include "utils.h"
#include "../utils/module1.h"

using namespace std;

// Define the Kernel function pointer type
typedef void (*KernelFunc)(const uint64_t* __restrict__, uint32_t* __restrict__, size_t, uint32_t, uint32_t);

// ==========================================
// DATA STRUCTURE FOR SUMMARY TABLE
// ==========================================
struct BenchmarkResult {
    string sScenario;
    string sAlgorithm;
    double dMedian_ms;
    double dStd_dev_ms;
    double dThroughput;
    double dMax_load_percent; 
};

// ==========================================
// BENCHMARK EXECUTION FUNCTION
// ==========================================
BenchmarkResult run_benchmark(const string& p_sScenario,
                              const string& p_sName, 
                              KernelFunc p_fKernel, 
                              const vector<uint64_t>& p_vKeys, 
                              vector<no_init_t<uint32_t>>& p_vPart_id, 
                              uint32_t p_uP, 
                              uint32_t p_uShift_val) 
{
    // ==========================================
    // 1. VARIABLE DECLARATIONS (Init at the top)
    // ==========================================
    size_t              l_stN;
    int                 l_iRUNS;
    vector<double>      l_vdTimes_ms;
    const uint64_t* __restrict__ l_pKeys;
    uint32_t* __restrict__ l_pPart_id;
    int                 l_i;
    double              l_dMedian_ms;
    double              l_dSum;
    double              l_dMean;
    double              l_dSq_sum;
    double              l_dStd_dev;
    double              l_dThroughput;
    volatile uint32_t   l_uDummy;
    BenchmarkResult     l_stResult;
    
    
    vector<size_t>      l_vHistogram;
    size_t              l_stMax_elements;
    double              l_dMax_load_percent;
    size_t              l_stJ;

    // ==========================================
    // 2. INITIALIZATION
    // ==========================================
    l_stN = p_vKeys.size();
    l_iRUNS = 10;
    l_vdTimes_ms.resize(l_iRUNS);
    l_pKeys = p_vKeys.data();
    l_pPart_id = (uint32_t*)p_vPart_id.data();
    l_vHistogram.resize(p_uP, 0);

    cout << "\nAlgorithm Test: " << p_sName << "\n";
    cout << string(40, '-') << "\n";

    
    std::fill(p_vPart_id.begin(), p_vPart_id.end(), 0);

    // ==========================================
    // 3. BENCHMARK EXECUTION (Using Prof's Timer)
    // ==========================================
    for (l_i = 0; l_i < l_iRUNS; ++l_i) {
        
        // Optional Warm-up for instruction/data cache
        if (l_i == 0) p_fKernel(l_pKeys, l_pPart_id, 1000, p_uP, p_uShift_val);

        TIMERSTART(Kernel)
        
        // KERNEL CALL - The loop over N elements happens entirely inside here
        p_fKernel(l_pKeys, l_pPart_id, l_stN, p_uP, p_uShift_val);
        
        TIMERSTOP(Kernel)
        
        // Save execution time in milliseconds for final statistics
        l_vdTimes_ms[l_i] = elapsed_Kernel * 1000.0;
    }

    // ==========================================
    // 4. QUALITY CHECK: DISTRIBUTION (HISTOGRAM)
    // ==========================================
    for (l_stJ = 0; l_stJ < l_stN; ++l_stJ) {
        l_vHistogram[l_pPart_id[l_stJ]]++;
    }
    
    l_stMax_elements = *max_element(l_vHistogram.begin(), l_vHistogram.end());
    l_dMax_load_percent = ((double)l_stMax_elements / l_stN) * 100.0;

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

    // ANTI-OPTIMIZATION TRICK
    l_uDummy = l_pPart_id[l_stN / 2]; 
    (void)l_uDummy;

    // ==========================================
    // 6. STORE AND RETURN RESULTS
    // ==========================================
    cout << "=> RESULT [" << p_sName << "]: "
         << "Median = " << fixed << setprecision(2) << l_dMedian_ms << " ms | "
         << "Throughput = " << l_dThroughput << " Mkeys/s | "
         << "Max Load = " << l_dMax_load_percent << " %\n\n";

    l_stResult.sScenario         = p_sScenario;
    l_stResult.sAlgorithm        = p_sName;
    l_stResult.dMedian_ms        = l_dMedian_ms;
    l_stResult.dStd_dev_ms       = l_dStd_dev;
    l_stResult.dThroughput       = l_dThroughput;
    l_stResult.dMax_load_percent = l_dMax_load_percent;

    return l_stResult;
}

int main() {
    // ==========================================
    // 1. LOCAL VARIABLE DECLARATIONS (Main)
    // ==========================================
    size_t l_stN;
    uint32_t l_uP;
    uint64_t l_uSeed;
    uint32_t l_uShift_val;
    size_t l_stI;
    
    vector<no_init_t<uint32_t>> l_vPart_id;
    vector<uint64_t> l_vKeys_uniform;
    vector<uint64_t> l_vKeys_dupes;
    vector<uint64_t> l_vKeys_pattern;
    vector<BenchmarkResult> l_vResults;

    // ==========================================
    // 2. ASSIGNMENTS AND SETUP
    // ==========================================
    l_stN = 50'000'000; 
    l_uP = 256;
    l_uSeed = 42;
    l_uShift_val = 64 - __builtin_ctz(l_uP);

    cout << "Allocating memory (" << l_stN << " elements)...\n";
    l_vPart_id.resize(l_stN);

    // ==========================================
    // SCENARIO 1: UNIFORM
    // ==========================================
    cout << "\n*** SCENARIO 1: Uniform (Key Space = UINT64_MAX) ***\n";
    l_vKeys_uniform = generate_keys(l_stN, l_uSeed, UINT64_MAX);
    
    l_vResults.push_back(run_benchmark("1: Uniform", "Modulo (% P)", kernel_modulo, l_vKeys_uniform, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("1: Uniform", "Bitmask (& P-1)", kernel_bitmask, l_vKeys_uniform, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("1: Uniform", "Shift+XOR", kernel_shift_xor, l_vKeys_uniform, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("1: Uniform", "Multiplicative", kernel_fibonacci, l_vKeys_uniform, l_vPart_id, l_uP, l_uShift_val));

    // ==========================================
    // SCENARIO 2: HIGH DUPLICATES
    // ==========================================
    cout << "\n*** SCENARIO 2: High Duplicates (Key Space = 1000) ***\n";
    l_vKeys_dupes = generate_keys(l_stN, l_uSeed, 1000);
    
    l_vResults.push_back(run_benchmark("2: High Dupl.", "Modulo (% P)", kernel_modulo, l_vKeys_dupes, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("2: High Dupl.", "Bitmask (& P-1)", kernel_bitmask, l_vKeys_dupes, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("2: High Dupl.", "Shift+XOR", kernel_shift_xor, l_vKeys_dupes, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("2: High Dupl.", "Multiplicative", kernel_fibonacci, l_vKeys_dupes, l_vPart_id, l_uP, l_uShift_val));

    // ==========================================
    // SCENARIO 3: MALICIOUS PATTERN (1024x)
    // ==========================================
    cout << "\n*** SCENARIO 3: Pattern (1024x) ***\n";
    
    l_vKeys_pattern = l_vKeys_uniform; 
    for(l_stI = 0; l_stI < l_stN; ++l_stI) {
        l_vKeys_pattern[l_stI] *= 1024; 
    }
    
    l_vResults.push_back(run_benchmark("3: Pattern", "Modulo (% P)", kernel_modulo, l_vKeys_pattern, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("3: Pattern", "Bitmask (& P-1)", kernel_bitmask, l_vKeys_pattern, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("3: Pattern", "Shift+XOR", kernel_shift_xor, l_vKeys_pattern, l_vPart_id, l_uP, l_uShift_val));
    l_vResults.push_back(run_benchmark("3: Pattern", "Multiplicative", kernel_fibonacci, l_vKeys_pattern, l_vPart_id, l_uP, l_uShift_val));

    // ==========================================
    // 6. PRINT SUMMARY TABLE
    // ==========================================
    cout << "\n\n";
    cout << "========================================================================================================\n";
    cout << "                                  SUMMARY TABLE (N = 50M, P = 256)                                     \n";
    cout << "========================================================================================================\n";
    cout << left << setw(18) << "Scenario" 
         << setw(18) << "Hash Function" 
         << right << setw(15) << "Time (ms)" 
         << setw(15) << "Std Dev (ms)" 
         << setw(20) << "Throughput (Mkeys/s)"
         << setw(15) << "Max Load (%)\n"; 
    cout << string(104, '-') << "\n";
    
    for (const auto& l_stRes : l_vResults) {
        cout << left << setw(18) << l_stRes.sScenario 
             << setw(18) << l_stRes.sAlgorithm 
             << right << setw(15) << fixed << setprecision(2) << l_stRes.dMedian_ms 
             << setw(15) << l_stRes.dStd_dev_ms 
             << setw(20) << l_stRes.dThroughput 
             << setw(14) << l_stRes.dMax_load_percent << "%\n"; 
    }
    cout << "========================================================================================================\n";

    return 0;
}