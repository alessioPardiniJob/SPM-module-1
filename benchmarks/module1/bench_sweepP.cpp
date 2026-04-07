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

#include "hpc_helpers.hpp"
#include "utils.h"
#include "module1.h"

using namespace std;

int main(int argc, char** argv) {
    // ==========================================
    // 1. DICHIARAZIONE VARIABILI
    // ==========================================
    size_t              l_stN;
    uint64_t            l_uSeed;
    int                 l_iRUNS;
    vector<double>      l_vdTimes_ms;
    int                 l_i;
    
    double              l_dMedian_ms;
    double              l_dSum, l_dMean, l_dSq_sum, l_dStd_dev, l_dThroughput;
    volatile uint16_t   l_uDummy;
    
    // Vettori
    vector<uint64_t>            l_vKeys;
    vector<no_init_t<uint16_t>> l_vPart_id;

    // ==========================================
    // 2. SETUP E ALLOCAZIONE
    // ==========================================
    l_stN = 50'000'000; 
    l_uSeed = 42;
    l_iRUNS = 10;
    
    l_vdTimes_ms.resize(l_iRUNS); 

    cout << "========================================================\n";
    cout << "           ESECUZIONE BENCHMARK (N = 50M)              \n";
    cout << "========================================================\n";

    cout << "[*] Generazione chiavi e allocazione memoria in corso...\n";
    l_vKeys = generate_keys(l_stN, l_uSeed, UINT64_MAX); 
    l_vPart_id.resize(l_stN);

    if ((reinterpret_cast<uintptr_t>(l_vKeys.data()) % 32 != 0) || 
        (reinterpret_cast<uintptr_t>(l_vPart_id.data()) % 32 != 0)) {
        cout << "[!] ATTENZIONE: Memoria non allineata a 32 byte!\n";
    }

    // ==========================================
    // NUOVO: CICLO PER SWEEP DI P (2, 16, 256, 4096, 65536)
    // ==========================================
    vector<uint32_t> p_values = {2, 16, 256, 4096, 65536};

    for (uint32_t l_uP : p_values) {
        cout << "\n\n========================================================\n";
        cout << " >>> TEST SWEEP P: " << l_uP << " PARTIZIONI <<<\n";
        cout << "========================================================\n";

        // ==========================================
        // TEST A: INTRINSICS (CPU)
        // ==========================================
        cout << "\n[--- 1. ESECUZIONE CPU (INTRINSICS) ---]\n";
        std::fill(l_vPart_id.begin(), l_vPart_id.end(), 0);
        compute_partitions(l_vKeys.data(), (uint16_t*)l_vPart_id.data(), 1000, l_uP); // Warm-up

        for (l_i = 0; l_i < l_iRUNS; ++l_i) {
            TIMERSTART(Kernel_CPU)
            compute_partitions(l_vKeys.data(), (uint16_t*)l_vPart_id.data(), l_stN, l_uP);
            TIMERSTOP(Kernel_CPU)
            
            // nvcc definisce __CUDACC__, quindi la macro crea timeKernel_CPU ed è già in ms!
            l_vdTimes_ms[l_i] = timeKernel_CPU; 
        }

        // Calcolo Statistiche CPU
        sort(l_vdTimes_ms.begin(), l_vdTimes_ms.end());
        l_dMedian_ms = (l_vdTimes_ms[l_iRUNS/2 - 1] + l_vdTimes_ms[l_iRUNS/2]) / 2.0;
        l_dSum = accumulate(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), 0.0);
        l_dMean = l_dSum / l_iRUNS;
        l_dSq_sum = inner_product(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), l_vdTimes_ms.begin(), 0.0);
        l_dStd_dev = sqrt(l_dSq_sum / l_iRUNS - l_dMean * l_dMean);
        l_dThroughput = (l_stN / (l_dMedian_ms / 1000.0)) / 1000000.0;

        cout << "   - Mediana: " << fixed << setprecision(2) << l_dMedian_ms << " ms\n";
        cout << "   - Std Dev: " << l_dStd_dev << " ms\n";
        cout << "   - T-put  : " << l_dThroughput << " Mkeys/s\n";
        
        uint64_t checksum_cpu = calculate_checksum((uint16_t*)l_vPart_id.data(), l_stN);
        cout << "   => CHECKSUM CPU: 0x" << hex << checksum_cpu << dec << "\n";

        // ==========================================
        // TEST B: CUDA (GPU)
        // ==========================================
        cout << "\n[--- 2. ESECUZIONE GPU (CUDA) ---]\n";
        std::fill(l_vPart_id.begin(), l_vPart_id.end(), 0);
        
        float t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;

        for (l_i = 0; l_i < l_iRUNS; ++l_i) {
            TIMERSTART(Total_GPU)
            
            // Passiamo le variabili per raccogliere i tempi specifici
            compute_partitions_cuda(l_vKeys.data(), (uint16_t*)l_vPart_id.data(), l_stN, l_uP, t_h2d, t_kernel, t_d2h);
            
            TIMERSTOP(Total_GPU)
            
            // Memorizziamo SOLO il tempo del kernel per calcolare le statistiche richieste dal prof
            l_vdTimes_ms[l_i] = t_kernel;
        }

        // Calcolo Statistiche GPU
        sort(l_vdTimes_ms.begin(), l_vdTimes_ms.end());
        l_dMedian_ms = (l_vdTimes_ms[l_iRUNS/2 - 1] + l_vdTimes_ms[l_iRUNS/2]) / 2.0;
        l_dSum = accumulate(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), 0.0);
        l_dMean = l_dSum / l_iRUNS;
        l_dSq_sum = inner_product(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), l_vdTimes_ms.begin(), 0.0);
        l_dStd_dev = sqrt(l_dSq_sum / l_iRUNS - l_dMean * l_dMean);
        l_dThroughput = (l_stN / (l_dMedian_ms / 1000.0)) / 1000000.0;

        cout << "   - Mediana (Kernel-only): " << fixed << setprecision(2) << l_dMedian_ms << " ms\n";
        cout << "   - Std Dev (Kernel-only): " << l_dStd_dev << " ms\n";
        cout << "   - T-put   (Kernel-only): " << l_dThroughput << " Mkeys/s\n";
        cout << "   [Breakdown Ultima Run ] H2D: " << t_h2d << " ms | Kernel: " << t_kernel << " ms | D2H: " << t_d2h << " ms\n";

        uint64_t checksum_gpu = calculate_checksum((uint16_t*)l_vPart_id.data(), l_stN);
        cout << "   => CHECKSUM GPU: 0x" << hex << checksum_gpu << dec << "\n";
        if (checksum_cpu != checksum_gpu) cout << "   [!] ERRORE: I CHECKSUM NON COINCIDONO!\n";

        // TRUCCO ANTI-OTTIMIZZAZIONE
        l_uDummy = ((uint16_t*)l_vPart_id.data())[l_stN / 2]; 
        (void)l_uDummy;
    }

    return 0;
}