CXX = c++
MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS = -fopenmp
LDLIBS =
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17
CUDA_ARCH =
BIN = bin
.PHONY: all cpu gpu clean test test-gpu
all: cpu
cpu: $(BIN)/pagerank_reference $(BIN)/power_symmetric $(BIN)/power_symmetric_mpi $(BIN)/pagerank $(BIN)/pagerank_mpi $(BIN)/pagerank_hybrid $(BIN)/make_graph $(BIN)/graph_stats $(BIN)/lanczos $(BIN)/lanczos_mpi
$(BIN):
	mkdir -p $@
$(BIN)/pagerank: baseline_a/main.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/pagerank_mpi: baseline_a/pagerank_mpi.cpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/pagerank_hybrid: src/pagerank_hybrid.cpp src/graph.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/make_graph: src/make_graph.cpp src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/graph_stats: src/graph_stats.cpp src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/lanczos: src/lanczos.cpp src/symmetric.hpp src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/lanczos_mpi: src/lanczos.cpp src/symmetric.hpp src/graph.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DLANCZOS_MPI $< -o $@ $(LDLIBS)
test: all
	python3 scripts/validate_lanczos.py
	python3 scripts/validate_eigen_pair.py
	python3 scripts/validate_gpu.py --reference-only --out results/cpu_reference_validation
clean:
	rm -rf bin

$(BIN)/pagerank_reference: src/pagerank_reference.cpp src/pagerank_common.hpp src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/power_symmetric: src/power_symmetric.cpp src/symmetric.hpp src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)
$(BIN)/power_symmetric_mpi: src/power_symmetric.cpp src/symmetric.hpp src/graph.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DLANCZOS_MPI $< -o $@ $(LDLIBS)
gpu: $(BIN)/pagerank_cuda
$(BIN)/pagerank_cuda: src/pagerank_cuda.cu src/pagerank_common.hpp src/graph.hpp | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH)) $< -o $@

test-gpu:
	bash scripts/gpu_preflight.sh


# Optional GPU Lanczos module; existing PageRank targets remain separate.
.PHONY: gpu-lanczos
gpu-lanczos:
	$(MAKE) -C gpu_lanczos BIN="$(abspath $(BIN))/gpu_lanczos" CXX="$(CXX)" MPICXX="$(MPICXX)" NVCC="$(NVCC)" CUDA_ARCH="$(if $(CUDA_ARCH),$(CUDA_ARCH),sm_86)"
