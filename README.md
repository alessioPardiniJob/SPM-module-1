
# Partitioned Hash Join Mapping — SPM Module 1

## Overview

This module implements and optimizes the **mapping phase** of a *Partitioned Hash Join*, a workload characterized by:

- **Low computational intensity**
- **High memory pressure (memory-bound)**
- Throughput limited by **DRAM bandwidth**

The project explores multiple optimization strategies:

```

.
├── Makefile
├── src/
│   ├── AVX2versions/
│   ├── module1_plain_16.cpp
│   ├── module1_plain_32.cpp
│   ├── module1_plain_64.cpp
│   ├── module1_avx2.cpp
│   ├── module1_cuda.cu
│   └── utils.cpp
├── benchmarks/module1/
├── utils/
├── bin/
├── assembly_report.txt
├── vec_report.txt

````

---

# EXPERIMENT 0 — Algorithm Selection (Scalar Baseline)

## HOW TO COMPILE

```bash
make mod1_algorithms
````

### Compilation Flags

```bash
-O3 -fno-tree-vectorize
```

#### `-O3`

* Enables aggressive optimizations:

  * loop unrolling
  * inlining
  * strength reduction
* Improves Instruction-Level Parallelism (ILP)
* Provides a realistic optimized scalar baseline

#### `-fno-tree-vectorize`

* Disables GCC auto-vectorization
* Guarantees purely scalar execution
* Prevents compiler-generated SIMD transformations

👉 Goal: isolate pure algorithmic cost without SIMD interference

---

## HOW TO EXECUTE

```bash
./bin/mod1_algorithms
```

### Output

* Throughput (Mkeys/s)
* Latency (ms)

---

# EXPERIMENT 1 — Data Width Sensitivity (16-bit vs 32-bit)

## HOW TO COMPILE

### Scalar baseline

```bash
make mod1_baseline_16
make mod1_baseline_32
```

### Auto-vectorized versions

```bash
make mod1_autovec_16
make mod1_autovec_32
```

---

### Compilation Flags

#### Scalar

```bash
-O3 -fno-tree-vectorize
```

#### Auto-vectorized

```bash
-O3 -ftree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt
```

---

### Flag Explanation

#### `-ftree-vectorize`

* Enables GCC loop vectorizer
* Converts scalar loops into SIMD operations
* Depends on:

  * dependency analysis
  * internal cost model

#### `-mavx2`

* Targets AVX2 ISA (256-bit SIMD)
* Enables:

  * YMM registers
  * operations on 8×32-bit or 16×16-bit lanes

#### `-fopt-info-vec-all=vec_report.txt`

* Generates a detailed vectorization report:

  * which loops were vectorized
  * which were not
  * reasons (dependencies, cost, alignment)

---

### Key Insight

* 16-bit reduces memory footprint
* However, narrowing (32→16) introduces expensive shuffle operations
* Leads to **Vectorization Paradox**: less memory ≠ better performance

---

## HOW TO EXECUTE

```bash
./bin/mod1_baseline_16
./bin/mod1_baseline_32
./bin/mod1_autovec_16
./bin/mod1_autovec_32
```

---

# EXPERIMENT 2 — Manual AVX2 Optimization

## HOW TO COMPILE

```bash
make mod1_naive_avx2
make mod1_vectorized_storage_avx2
```

---

### Compilation Flags

```bash
-O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt
```

---

### Flag Explanation

#### `-fno-tree-vectorize`

* Disables compiler auto-vectorization
* Prevents interference with manual SIMD intrinsics

#### `-mavx2`

* Enables AVX2 instruction set
* Required for `_mm256_*` intrinsics

---

### Implementation Differences

#### Naive AVX2

* SIMD computation
* Scalar extraction for storage
* Overhead:

  * domain crossing (SIMD ↔ scalar)

#### Vectorized Storage

* Fully SIMD pipeline
* Uses:

  ```cpp
  _mm256_shuffle_ps
  ```
* Eliminates scalar bottleneck

👉 Result: higher throughput due to full SIMD utilization

---

## HOW TO EXECUTE

```bash
./bin/mod1_naive_avx2
./bin/mod1_vectorized_storage_avx2
```

---

# EXPERIMENT 3 — Memory Optimization (Streaming Stores + Prefetch)

## HOW TO COMPILE

### Streaming stores

```bash
make mod1_vectorized_streamingStore_avx2
```

### Prefetch tuning

```bash
make mod1_vectorized_streamingPrefetch16_avx2
make mod1_vectorized_streamingPrefetch32_avx2
make mod1_vectorized_streamingPrefetch64_avx2
make mod1_vectorized_streamingPrefetch128_avx2
```

---

### Compilation Flags

```bash
-O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt
```

---

### Technical Details

#### Streaming Stores

```cpp
_mm256_stream_si256(...)
```

* Bypasses L1/L2 caches
* Writes directly to memory (write-combining buffers)
* Avoids:

  * cache pollution
  * write-allocate traffic

⚠️ Requirements:

* 32-byte aligned memory
* scalar prologue for alignment

---

#### Software Prefetching

```cpp
_mm_prefetch(ptr, _MM_HINT_T0)
```

* Fetches data into cache before use
* Implements **latency hiding**
* Reduces stall cycles due to memory access

---

### Critical Parameter: Prefetch Distance

| Distance  | Effect             |
| --------- | ------------------ |
| Too small | Latency not hidden |
| Too large | Cache eviction     |

👉 Optimal (empirical):

```
+64 keys ≈ 512 bytes
```

---

## HOW TO EXECUTE

```bash
./bin/mod1_vectorized_streamingStore_avx2
./bin/mod1_vectorized_streamingPrefetch64_avx2
```

---

# EXPERIMENT 4 — GPU Offloading (CUDA)

## HOW TO COMPILE

⚠️ Run on a GPU-enabled node (e.g., `gpu-excl`)

```bash
make mod1_cuda
```

---

### Compilation Flags

```bash
nvcc -O3 -std=c++17 -arch=native
```

---

### Flag Explanation

#### `-O3`

* Enables aggressive optimization for CUDA kernels

#### `-arch=native`

* Targets the local GPU architecture
* Alternative:

  ```bash
  -arch=sm_70
  -arch=sm_80
  ```

---

### Execution Model

1. Host → Device transfer (`cudaMemcpy`)
2. Kernel execution
3. Device → Host transfer

👉 Dominant bottleneck:

```
PCIe bandwidth
```

---

## HOW TO EXECUTE

```bash
./bin/mod1_cuda
```

### Output

* Kernel execution time
* Memory transfer time
* Total runtime

👉 Typical behavior:

* Kernel extremely fast
* `cudaMemcpy` dominates (~90%+)

---

# EXPERIMENT 5 — Sensitivity Analysis (Sweep P)

## HOW TO COMPILE

```bash
make mod1_sweepP
```

---

### Compilation Flags

```bash
nvcc -O3 -std=c++17 -arch=native -Xcompiler "-mavx2"
```

---

### Flag Explanation

#### `-Xcompiler "-mavx2"`

* Forwards AVX2 flag to host compiler (g++)
* Required for CPU-side SIMD code inside CUDA build

---

## HOW TO EXECUTE

```bash
./bin/mod1_sweepP
```

---

### Parameters

* Number of partitions:

```
P ∈ [2, 65536]
```

---

### Expected Behavior

* Flat performance curve
* Mapping operation:

  ```
  h(k) & (P - 1)
  ```
* Constant cost independent of P

👉 Limitation:

```
memory bandwidth bound
```

---

# Key Performance Insights

* The workload is strictly **memory-bound**
* SIMD improves compute throughput but does not remove DRAM bottleneck
* Streaming stores reduce unnecessary cache traffic
* Prefetching hides memory latency
* CUDA is inefficient for this workload:

  * low compute intensity
  * high PCIe overhead

---

# Full Build

```bash
make all -j
```

---

# Executables

All binaries are located in:

```bash
bin/
```

---

# Cleanup

```bash
make clean
make cleanall
```

```
```
