// Blindiamo il file di test: NIENTE vettorizzazione qui dentro.
// Vogliamo misurare solo il kernel, non il codice di contorno.
#ifndef __CUDACC__
    #pragma GCC optimize("O3,no-tree-vectorize")
#endif
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <string>
#include <chrono>

#include "hpc_helpers.hpp"
#include "utils.h"
#include "module1.h"
#ifndef __CUDACC__
    #pragma GCC optimize("O3,no-tree-vectorize")
#endif

#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <string>
#include <chrono>

#include "utils.h"
#include "module1.h"

using namespace std;

// Supporto per la mediana
double get_median(vector<double>& v) {
    sort(v.begin(), v.end());
    size_t n = v.size();
    return (v[n/2 - 1] + v[n/2]) / 2.0;
}

// Supporto per la Deviazione Standard
double get_std_dev(const vector<double>& v) {
    double sum = accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = inner_product(v.begin(), v.end(), v.begin(), 0.0);
    return sqrt(sq_sum / v.size() - mean * mean);
}

int main(int argc, char** argv) {
    size_t              l_stN = 50'000'000; 
    uint32_t            l_uP = 4096;
    uint64_t            l_uSeed = 42;
    int                 l_iRUNS = 10;
    
    vector<double>      l_vdTimes_H2D(l_iRUNS);
    vector<double>      l_vdTimes_Kernel(l_iRUNS);
    vector<double>      l_vdTimes_D2H(l_iRUNS);
    vector<double>      l_vdTimes_Total(l_iRUNS);
    
    volatile uint16_t   l_uDummy;
    vector<uint64_t>            l_vKeys;
    vector<no_init_t<uint16_t>> l_vPart_id;

    cout << "========================================================\n";
    cout << "           ESECUZIONE BENCHMARK CUDA (N = 100M)         \n";
    cout << "========================================================\n";

    cout << "[*] Generazione chiavi e allocazione memoria in corso...\n";
    l_vKeys = generate_keys(l_stN, l_uSeed, UINT64_MAX); 
    l_vPart_id.resize(l_stN);
    std::fill(l_vPart_id.begin(), l_vPart_id.end(), 0);

    // Warm-up dummy
    float dummy_h2d, dummy_kernel, dummy_d2h;
    compute_partitions_cuda(l_vKeys.data(), (uint16_t*)l_vPart_id.data(), 1000, l_uP, dummy_h2d, dummy_kernel, dummy_d2h);

    cout << "[*] Esecuzione su GPU in corso (" << l_iRUNS << " run)...\n";
    
    for (int l_i = 0; l_i < l_iRUNS; ++l_i) {
        float t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;
        
        auto start_total = std::chrono::steady_clock::now();
        compute_partitions_cuda(l_vKeys.data(), (uint16_t*)l_vPart_id.data(), l_stN, l_uP, t_h2d, t_kernel, t_d2h);
        auto stop_total = std::chrono::steady_clock::now();
        
        double elapsed_total_ms = std::chrono::duration<double, std::milli>(stop_total - start_total).count();
        
        l_vdTimes_H2D[l_i]    = t_h2d;
        l_vdTimes_Kernel[l_i] = t_kernel;
        l_vdTimes_D2H[l_i]    = t_d2h;
        l_vdTimes_Total[l_i]  = elapsed_total_ms;
        
        // Stampa di progresso pulita (opzionale ma bella da vedere)
        cout << "   Run " << setw(2) << l_i + 1 << " | Kernel: " 
             << fixed << setprecision(3) << setw(6) << t_kernel << " ms | Total E2E: " 
             << setw(6) << elapsed_total_ms << " ms\n";
    }

    // Calcolo Mediane
    double med_h2d    = get_median(l_vdTimes_H2D);
    double med_kernel = get_median(l_vdTimes_Kernel);
    double med_d2h    = get_median(l_vdTimes_D2H);
    double med_total  = get_median(l_vdTimes_Total);

    // Calcolo Deviazioni Standard
    double std_h2d    = get_std_dev(l_vdTimes_H2D);
    double std_kernel = get_std_dev(l_vdTimes_Kernel);
    double std_d2h    = get_std_dev(l_vdTimes_D2H);
    double std_total  = get_std_dev(l_vdTimes_Total);

    double tput_kernel = (l_stN / (med_kernel / 1000.0)) / 1000000.0;
    double tput_total  = (l_stN / (med_total / 1000.0))  / 1000000.0;

    cout << "\n=> BREAKDOWN DEI TEMPI (Mediana su " << l_iRUNS << " run):\n";
    cout << "   - Host-to-Device (PCIe) : " << fixed << setprecision(2) << med_h2d << " ms (Std Dev: " << std_h2d << " ms)\n";
    cout << "   - Kernel Execution (GPU): " << med_kernel << " ms (Std Dev: " << std_kernel << " ms)\n";
    cout << "   - Device-to-Host (PCIe) : " << med_d2h << " ms (Std Dev: " << std_d2h << " ms)\n";
    cout << "   - Total End-to-End      : " << med_total << " ms (Std Dev: " << std_total << " ms)\n";
    cout << "--------------------------------------------------------\n";
    cout << "=> THROUGHPUT:\n";
    cout << "   - Puro Kernel GPU       : " << tput_kernel << " Mkeys/s\n";
    cout << "   - Totale End-to-End     : " << tput_total << " Mkeys/s\n";
    cout << "========================================================\n";

    l_uDummy = ((uint16_t*)l_vPart_id.data())[l_stN / 2]; 
    (void)l_uDummy;

    cout << "[*] Calcolo checksum per verifica...\n";
    uint64_t final_checksum = calculate_checksum((uint16_t*)l_vPart_id.data(), l_stN);
    cout << "=> CHECKSUM: 0x" << hex << final_checksum << dec << "\n";
    cout << "========================================================\n";

    return 0;
}