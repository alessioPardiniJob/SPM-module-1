
# Partitioned Hash Join Mapping — SPM Module 1

## Overview

This module implements and optimizes the **mapping phase** of a *Partitioned Hash Join*.

## Repository Structure

```

.
├── Makefile
├── src/
│   ├── AVX2versions/
│   │   ├── module1_naive_version_avx2.cpp
│   │   ├── module1_vectorized_storage_avx2.cpp
│   │   ├── module1_vectorized_streamingPrefetch16_avx2.cpp
│   │   ├── module1_vectorized_streamingPrefetch32_avx2.cpp
│   │   ├── module1_vectorized_streamingPrefetch64_avx2.cpp
│   │   ├── module1_vectorized_streamingPrefetch128_avx2.cpp
│   │   ├── module1_vectorized_streamingStore_avx2.cpp
│   ├── module1_plain_16.cpp
│   ├── module1_plain_32.cpp
│   ├── module1_plain_64.cpp
│   ├── module1_cuda.cu
│   └── utils.cpp
├── benchmarks/
│   ├── module1/
│   │   ├── bench_algorithms.cpp
│   │   ├── bench_module1_16.cpp
│   │   ├── bench_module1_32.cpp
│   │   ├── bench_module1_64.cpp
│   │   ├── bench_module1_cuda.cpp
│   │   ├── bench_sweepP.cpp
├── utils/
│   ├── hpc_helpers.hpp
│   ├── module1.h
│   ├── utils.h
├── bin/
├── vec_report.txt

````

---

# EXPERIMENT 0 — Algorithm Selection (Scalar Baseline)

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_algorithms
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -o bin/mod1_algorithms benchmarks/module1/bench_algorithms.cpp src/utils.cpp
```

#### `-fno-tree-vectorize`

👉 Goal: isolate pure algorithmic cost without SIMD interference

---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:01:00  ./bin/mod1_algorithms
```

### Output

Output to check: summary table reporting algorithm runtimes.

---

# EXPERIMENT 1 — Data Width Sensitivity (16-bit vs 32-bit)

## HOW TO COMPILE

### Scalar baseline

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_baseline_16
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_baseline_32
```


### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -o bin/mod1_baseline_16 benchmarks/module1/bench_module1_16.cpp src/module1_plain_16.cpp src/utils.cpp

g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -o bin/mod1_baseline_32 benchmarks/module1/bench_module1_32.cpp src/module1_plain_32.cpp src/utils.cpp
```

- `-fno-tree-vectorize` → disables vectorization to isolate pure algorithmic cost (no SIMD)

### Auto-vectorized versions

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_autovec_16
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_autovec_32
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -ftree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_autovec_16 benchmarks/module1/bench_module1_16.cpp src/module1_plain_16.cpp src/utils.cpp

g++ -std=c++17 -Wall -I./utils -O3 -ftree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_autovec_32 benchmarks/module1/bench_module1_32.cpp src/module1_plain_32.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-ftree-vectorize` → enables auto-vectorization
- `-mavx2` → use AVX2 SIMD instructions
- `-fopt-info-vec-all=vec_report.txt` → generates detailed vectorization report
---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_baseline_16
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_baseline_32
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_autovec_16
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_autovec_32
```
### Output

Output to check: for every run look at the section "=> STATISTICAL RESULTS"

---

# EXPERIMENT 2 — Manual AVX2 Optimization

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_naive_avx2
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_naive_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_naive_version_avx2.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-fno-tree-vectorize` → disable automatic vectorization
- `-mavx2` → use AVX2 SIMD instructions


---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_naive_avx2
```

### Output

Output to check: look at the section "=> STATISTICAL RESULTS"

---


# EXPERIMENT 3 — Vectorized Storage AVX2

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_storage_avx2
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_storage_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_storage_avx2.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-fno-tree-vectorize` → disable automatic vectorization
- `-mavx2` → use AVX2 SIMD instructions


---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_storage_avx2
```

### Output

Output to check: look at the section "=> STATISTICAL RESULTS"

---


# EXPERIMENT 4 — Streaming Stores 

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingStore_avx2
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingStore_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingStore_avx2.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-fno-tree-vectorize` → disable automatic vectorization
- `-mavx2` → use AVX2 SIMD instructions


---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingStore_avx2
```

### Output

Output to check: look at the section "=> STATISTICAL RESULTS"

---

# EXPERIMENT 4 — Streaming Stores 

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingStore_avx2
```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingStore_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingStore_avx2.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-fno-tree-vectorize` → disable automatic vectorization
- `-mavx2` → use AVX2 SIMD instructions


---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingStore_avx2
```

### Output

Output to check: look at the section "=> STATISTICAL RESULTS"

---


# EXPERIMENT 5 — Prefetching 

This experiment evaluated the effect of different prefetch distances and strategies on the partitioning routine’s performance. By prefetching data ahead of use, memory latency is reduced, allowing comparison of techniques to find the most efficient balance between cache use and computation speed.

## HOW TO COMPILE

```bash
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingPrefetch16_avx2
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingPrefetch32_avx2
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingPrefetch64_avx2
srun -p gpu-shared -w node09 --time=00:00:30 make mod1_vectorized_streamingPrefetch128_avx2

```

### Compilation Flags

```bash
g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingPrefetch16_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingPrefetch16_avx2.cpp src/utils.cpp

g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingPrefetch32_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingPrefetch32_avx2.cpp src/utils.cpp

g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingPrefetch64_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingPrefetch64_avx2.cpp src/utils.cpp

g++ -std=c++17 -Wall -I./utils -O3 -fno-tree-vectorize -mavx2 -fopt-info-vec-all=vec_report.txt -o bin/mod1_vectorized_streamingPrefetch128_avx2 benchmarks/module1/bench_module1_16.cpp src/AVX2versions/module1_vectorized_streamingPrefetch128_avx2.cpp src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-fno-tree-vectorize` → disable automatic vectorization
- `-mavx2` → use AVX2 SIMD instructions


---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingPrefetch16_avx2
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingPrefetch32_avx2
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingPrefetch64_avx2
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_vectorized_streamingPrefetch128_avx2
```

### Output

Output to check: for each run, look at the section "=> STATISTICAL RESULTS"

---


# EXPERIMENT 6 — CUDA (optional) 

## HOW TO COMPILE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 make mod1_cuda
```

### Compilation Flags

```bash
nvcc -O3 -std=c++17  -x cu -I./utils -arch=native -o bin/mod1_cuda benchmarks/module1/bench_module1_cuda.cpp src/module1_cuda.cu src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-arch=native` → compile for the host architecture (GPU/CPU specific code)

---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_cuda
```

### Output

Output to check: look at the section "=> TIME BREAKDOWN (Median over 10 runs):"

---

# EXPERIMENT 7 — Sensitivity Analysis (Sweep P)

* Number of partitions:

```
P ∈ [2, 65536]
```
## HOW TO COMPILE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 make mod1_sweepP
```

### Compilation Flags

```bash
nvcc -O3 -std=c++17  -x cu -I./utils -arch=native -Xcompiler "-mavx2" -o bin/mod1_sweepP benchmarks/module1/bench_sweepP.cpp src/AVX2versions/module1_vectorized_streamingPrefetch64_avx2.cpp src/module1_cuda.cu src/utils.cpp
```

- `-O3` → aggressive compiler optimizations
- `-arch=native` → compile for the host architecture (GPU/CPU specific code)

---

## HOW TO EXECUTE

```bash
srun -p gpu-excl -w node09 --time=00:00:30 ./bin/mod1_sweepP
```

### Output

Output to check: for every "TEST SWEEP P: x PARTITIONS", look at the section " CPU EXECUTION (INTRINSICS) " and "GPU EXECUTION (CUDA)".

---

