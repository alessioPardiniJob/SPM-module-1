// Blindiamo il file di test: NIENTE vettorizzazione qui dentro.
// Vogliamo misurare solo il kernel, non il codice di contorno.
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
    // 1. DICHIARAZIONE VARIABILI
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
    volatile uint32_t   l_uDummy;
    
    // Vettori
    vector<uint64_t>            l_vKeys;
    vector<no_init_t<uint32_t>> l_vPart_id;

    // ==========================================
    // 2. SETUP E ALLOCAZIONE
    // ==========================================
    l_stN = 50'000'000; 
    l_uP = 4096;
    l_uSeed = 42;
    l_iRUNS = 10;
    
    l_vdTimes_ms.resize(l_iRUNS); // Alloco spazio per i tempi

    cout << "========================================================\n";
    cout << "           ESECUZIONE BENCHMARK (N = 100M)              \n";
    cout << "========================================================\n";

    cout << "[*] Generazione chiavi e allocazione memoria in corso...\n";
    // ⚠️ FONDAMENTALE: Generare e ridimensionare PRIMA di fare warm-up
    l_vKeys = generate_keys(l_stN, l_uSeed, UINT64_MAX); 
    l_vPart_id.resize(l_stN);

    // Controllo allineamento per sicurezza AVX2
    if ((reinterpret_cast<uintptr_t>(l_vKeys.data()) % 32 != 0) || 
        (reinterpret_cast<uintptr_t>(l_vPart_id.data()) % 32 != 0)) {
        cout << "[!] ATTENZIONE: Memoria non allineata a 32 byte!\n";
    }

    // ==========================================
    // 3. WARM-UP & FIRST-TOUCH POLICY
    // ==========================================
    // Ora che l'array esiste, forziamo il S.O. ad allocare fisicamente 
    // le pagine di memoria azzerando tutto l'array di output.
    std::fill(l_vPart_id.begin(), l_vPart_id.end(), 0);
    
    // Piccolo warm-up per le cache istruzioni
    //compute_partitions(l_vKeys.data(), (uint32_t*)l_vPart_id.data(), 1000, l_uP);
    compute_partitions(l_vKeys.data(), (uint32_t*)l_vPart_id.data(), 1000, l_uP);


    // ==========================================
    // 4. CICLO DI BENCHMARK (10 RUN)
    // ==========================================
    cout << "[*] Esecuzione del Kernel (" << l_iRUNS << " run)...\n";
    
    for (l_i = 0; l_i < l_iRUNS; ++l_i) {
        
        TIMERSTART(Kernel)
        
        // LA CHIAMATA UNIVERSALE AL KERNEL
        compute_partitions(l_vKeys.data(), (uint32_t*)l_vPart_id.data(), l_stN, l_uP);
        //compute_partitions(l_vKeys.data(), (uint32_t*)l_vPart_id.data(), l_stN, l_uP);

        TIMERSTOP(Kernel)
        
        l_vdTimes_ms[l_i] = elapsed_Kernel * 1000.0;
    }

    // ==========================================
    // 5. CALCOLO STATISTICHE
    // ==========================================
    sort(l_vdTimes_ms.begin(), l_vdTimes_ms.end());
    l_dMedian_ms = (l_vdTimes_ms[l_iRUNS/2 - 1] + l_vdTimes_ms[l_iRUNS/2]) / 2.0;

    l_dSum = accumulate(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), 0.0);
    l_dMean = l_dSum / l_iRUNS;
    l_dSq_sum = inner_product(l_vdTimes_ms.begin(), l_vdTimes_ms.end(), l_vdTimes_ms.begin(), 0.0);
    l_dStd_dev = sqrt(l_dSq_sum / l_iRUNS - l_dMean * l_dMean);

    l_dThroughput = (l_stN / (l_dMedian_ms / 1000.0)) / 1000000.0;

    // ==========================================
    // 6. STAMPA RISULTATI
    // ==========================================
    cout << "\n=> RISULTATO STATISTICO:\n";
    cout << "   - Mediana: " << fixed << setprecision(2) << l_dMedian_ms << " ms\n";
    cout << "   - Std Dev: " << l_dStd_dev << " ms\n";
    cout << "   - T-put  : " << l_dThroughput << " Mkeys/s\n";
    cout << "========================================================\n";

    // TRUCCO ANTI-OTTIMIZZAZIONE
    l_uDummy = ((uint32_t*)l_vPart_id.data())[l_stN / 2]; 
    (void)l_uDummy;

    //l_uDummy = ((uint32_t*)l_vPart_id.data())[l_stN / 2]; 
    //(void)l_uDummy;

    return 0;
}