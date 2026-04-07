#include "module1.h"
#include "hpc_helpers.hpp"
#include <cuda_runtime.h>
#include <iostream>

// ====================================================================
// KERNEL DEVICE: Eseguito sulla GPU, calcola un singolo elemento
// ====================================================================
__global__ void compute_partitions_kernel(const uint64_t* __restrict__ d_keys, 
                                          uint16_t* __restrict__ d_part_id, 
                                          size_t N, 
                                          uint32_t mask) 
{
    // 1. Calcolo dell'indice globale del thread 1D
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Controllo dei limiti (Boundary check vitale per l'ultimo blocco)
    if (i < N) {
        uint64_t k = d_keys[i];
        // Logica scalare identica al C++ base, la GPU gestisce il parallelismo
        d_part_id[i] = static_cast<uint16_t>((k ^ (k >> 32))) & mask;
    }
}

// ====================================================================
// FUNZIONE HOST: Orchestra trasferimenti e lancio del kernel
// ====================================================================
// Aggiunti t_h2d, t_kernel, t_d2h passati per riferimento!
void compute_partitions_cuda(const uint64_t* h_keys, 
                             uint16_t* h_part_id, 
                             size_t N, 
                             uint32_t P,
                             float& t_h2d,
                             float& t_kernel,
                             float& t_d2h) 
{
    uint32_t mask = P - 1;
    
    uint64_t* d_keys = nullptr;
    uint16_t* d_part_id = nullptr;

    size_t bytes_keys = N * sizeof(uint64_t);
    size_t bytes_parts = N * sizeof(uint16_t);

    // 1. ALLOCAZIONE DEVICE
    CUERR { cudaMalloc((void**)&d_keys, bytes_keys); }
    CUERR { cudaMalloc((void**)&d_part_id, bytes_parts); }

    // 2. TRASFERIMENTO HOST-TO-DEVICE (H2D)
    TIMERSTART(H2D_Transfer)
    CUERR { cudaMemcpy(d_keys, h_keys, bytes_keys, H2D); }
    TIMERSTOP(H2D_Transfer)
    // "Rubiamo" la variabile generata dalla macro e la passiamo al main!
    t_h2d = timeH2D_Transfer; 

    // 3. CONFIGURAZIONE E LANCIO DEL KERNEL
    int threadsPerBlock = 256;
    size_t numBlocks = SDIV(N, threadsPerBlock);

    TIMERSTART(Kernel_Execution)
    compute_partitions_kernel<<<numBlocks, threadsPerBlock>>>(d_keys, d_part_id, N, mask);
    CUERR { cudaGetLastError(); } // Controlla errori sincroni nel lancio
    CUERR { cudaDeviceSynchronize(); } // Attende la fine del kernel
    TIMERSTOP(Kernel_Execution)
    // Passiamo il tempo del kernel al main
    t_kernel = timeKernel_Execution;

    // 4. TRASFERIMENTO DEVICE-TO-HOST (D2H)
    TIMERSTART(D2H_Transfer)
    CUERR { cudaMemcpy(h_part_id, d_part_id, bytes_parts, D2H); }
    TIMERSTOP(D2H_Transfer)
    // Passiamo il tempo di copia indietro al main
    t_d2h = timeD2H_Transfer;

    // 5. PULIZIA MEMORIA
    CUERR { cudaFree(d_keys); }
    CUERR { cudaFree(d_part_id); }
}