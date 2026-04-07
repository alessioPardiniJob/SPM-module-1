# ==========================================
# Makefile Globale - SPM (CORRETTO)
# ==========================================

CXX        = g++
CXXFLAGS   = -std=c++17 -Wall -I./utils
OPTFLAGS   = -O3

# Variabili per CUDA (Modulo Opzionale)
NVCC       = nvcc
NVCCFLAGS  = -O3 -std=c++17  -x cu -I./utils -arch=native


VECREPORT = -fopt-info-vec-all=vec_report.txt

# Directory Strutturali
SRC_DIR    = src
BENCH_DIR  = benchmarks
BIN_DIR    = bin

# Sottocartelle Moduli
MOD1_BENCH = $(BENCH_DIR)/module1
MOD1_SRC   = $(SRC_DIR)

# Core Utils
UTILS_SRC  = $(SRC_DIR)/utils.cpp

# Eseguibili del module 1
MOD1_TARGETS = $(BIN_DIR)/mod1_algorithms \
               $(BIN_DIR)/mod1_baseline_32 \
               $(BIN_DIR)/mod1_baseline_16 \
               $(BIN_DIR)/mod1_autovec_32 \
               $(BIN_DIR)/mod1_autovec_16 \
               $(BIN_DIR)/mod1_naive_avx2 \
			   $(BIN_DIR)/mod1_vectorized_storage_avx2 \
			   $(BIN_DIR)/mod1_vectorized_streamingStore_avx2 \
			   $(BIN_DIR)/mod1_vectorized_streamingPrefetch16_avx2 \
			   $(BIN_DIR)/mod1_vectorized_streamingPrefetch32_avx2 \
			   $(BIN_DIR)/mod1_vectorized_streamingPrefetch64_avx2 \
			   $(BIN_DIR)/mod1_vectorized_streamingPrefetch128_avx2 \
			   $(BIN_DIR)/mod1_cuda \
			   $(BIN_DIR)/mod1_sweepP \

.PHONY: all clean cleanall directories mod1 mod1_algorithms mod1_baseline_32 mod1_baseline_16 mod1_autovec_32 mod1_autovec_16 mod1_naive_avx2 mod1_vectorized_storage_avx2 mod1_vectorized_streamingStore_avx2 mod1_vectorized_streamingPrefetch16_avx2 mod1_vectorized_streamingPrefetch32_avx2 mod1_vectorized_streamingPrefetch64_avx2 mod1_vectorized_streamingPrefetch128_avx2 mod1_cuda mod1_sweepP

all: directories $(MOD1_TARGETS)

mod1: directories $(MOD1_TARGETS)

mod1_algorithms: directories $(BIN_DIR)/mod1_algorithms
mod1_baseline_64:   directories $(BIN_DIR)/mod1_baseline_64
mod1_baseline_32:   directories $(BIN_DIR)/mod1_baseline_32
mod1_baseline_16:   directories $(BIN_DIR)/mod1_baseline_16
mod1_autovec_64:    directories $(BIN_DIR)/mod1_autovec_64
mod1_autovec_32:    directories $(BIN_DIR)/mod1_autovec_32
mod1_autovec_16:    directories $(BIN_DIR)/mod1_autovec_16
mod1_naive_avx2:    directories $(BIN_DIR)/mod1_naive_avx2
mod1_vectorized_storage_avx2: directories $(BIN_DIR)/mod1_vectorized_storage_avx2
mod1_vectorized_streamingStore_avx2: directories $(BIN_DIR)/mod1_vectorized_streamingStore_avx2
mod1_vectorized_streamingPrefetch16_avx2: directories $(BIN_DIR)/mod1_vectorized_streamingPrefetch16_avx2
mod1_vectorized_streamingPrefetch32_avx2: directories $(BIN_DIR)/mod1_vectorized_streamingPrefetch32_avx2
mod1_vectorized_streamingPrefetch64_avx2: directories $(BIN_DIR)/mod1_vectorized_streamingPrefetch64_avx2
mod1_vectorized_streamingPrefetch128_avx2: directories $(BIN_DIR)/mod1_vectorized_streamingPrefetch128_avx2
mod1_cuda: directories $(BIN_DIR)/mod1_cuda
mod1_sweepP: directories $(BIN_DIR)/mod1_sweepP


directories:
	@mkdir -p $(BIN_DIR)

# ======================================================================
# REGOLE DI COMPILAZIONE - module 1
# ======================================================================

# 0. Laboratorio Algoritmi
$(BIN_DIR)/mod1_algorithms: $(MOD1_BENCH)/bench_algorithms.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -o $@ $^

$(BIN_DIR)/mod1_baseline_64: $(MOD1_BENCH)/bench_module1_64.cpp $(MOD1_SRC)/module1_plain_64.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -o $@ $^

$(BIN_DIR)/mod1_autovec_64: $(MOD1_BENCH)/bench_module1_64.cpp $(MOD1_SRC)/module1_plain_64.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -ftree-vectorize -mavx2 $(VECREPORT) -o $@ $^

# 1. Baseline Scalare 32-bit
$(BIN_DIR)/mod1_baseline_32: $(MOD1_BENCH)/bench_module1_32.cpp $(MOD1_SRC)/module1_plain_32.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -o $@ $^

# 2. Auto-Vettorizzazione GCC 32-bit
$(BIN_DIR)/mod1_autovec_32: $(MOD1_BENCH)/bench_module1_32.cpp $(MOD1_SRC)/module1_plain_32.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -ftree-vectorize -mavx2 $(VECREPORT) -o $@ $^

# 1. Baseline Scalare 16-bit
$(BIN_DIR)/mod1_baseline_16: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/module1_plain_16.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -o $@ $^

# 2. Auto-Vettorizzazione GCC 16-bit
$(BIN_DIR)/mod1_autovec_16: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/module1_plain_16.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -ftree-vectorize -mavx2 $(VECREPORT) -o $@ $^






$(BIN_DIR)/mod1_naive_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_naive_version_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_storage_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_storage_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_streamingStore_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_streamingStore_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_streamingPrefetch16_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_streamingPrefetch16_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_streamingPrefetch32_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_streamingPrefetch32_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_streamingPrefetch64_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_streamingPrefetch64_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^

$(BIN_DIR)/mod1_vectorized_streamingPrefetch128_avx2: $(MOD1_BENCH)/bench_module1_16.cpp $(MOD1_SRC)/AVX2versions/module1_vectorized_streamingPrefetch128_avx2.cpp $(UTILS_SRC)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -fno-tree-vectorize -mavx2 $(VECREPORT) -o $@ $^


# ======================================================================
# REGOLE DI COMPILAZIONE - CUDA
# ======================================================================

# Regola per compilare il codice CUDA usando nvcc (Separate Compilation or Single Pass)
# Assumendo che tu scriva un bench_module1_cuda.cpp e un module1_cuda.cu
$(BIN_DIR)/mod1_cuda: $(MOD1_BENCH)/bench_module1_cuda.cpp $(MOD1_SRC)/module1_cuda.cu $(UTILS_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^


# ======================================================================
# REGOLE DI COMPILAZIONE - Sweep P
# ======================================================================
$(BIN_DIR)/mod1_sweepP: $(MOD1_BENCH)/bench_sweepP.cpp $(MOD1_SRC)/module1_avx2.cpp  $(MOD1_SRC)/module1_cuda.cu $(UTILS_SRC)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "-mavx2" -o $@ $^








