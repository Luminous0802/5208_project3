CXX = g++
MPICXX = mpicxx
NVCC = nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_LIBDIR ?= $(CUDA_HOME)/lib64
CUDA_ARCH ?= sm_86
BIN ?= bin
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic
ROOT = ..
.PHONY: all
all: $(BIN)/make_graph $(BIN)/lanczos_serial $(BIN)/lanczos_cpu_mpi $(BIN)/lanczos_cuda_mpi
$(BIN):
	mkdir -p $@
$(BIN)/make_graph: $(ROOT)/src/make_graph.cpp $(ROOT)/src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN)/lanczos_serial: $(ROOT)/src/lanczos.cpp $(ROOT)/src/symmetric.hpp $(ROOT)/src/graph.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@
$(BIN)/lanczos_cpu_mpi: $(ROOT)/src/lanczos.cpp $(ROOT)/src/symmetric.hpp $(ROOT)/src/graph.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) -fopenmp -DLANCZOS_MPI $< -o $@
$(BIN)/device_lanczos.o: src/device_lanczos.cu src/device_lanczos.hpp | $(BIN)
	$(NVCC) -ccbin $(CXX) -O3 -std=c++17 -arch=$(CUDA_ARCH) -c $< -o $@
$(BIN)/lanczos_cuda_mpi: src/lanczos_cuda.cpp src/device_lanczos.hpp $(ROOT)/src/symmetric.hpp $(ROOT)/src/graph.hpp $(BIN)/device_lanczos.o | $(BIN)
	$(MPICXX) $(CXXFLAGS) -fopenmp -DLANCZOS_MPI $< $(BIN)/device_lanczos.o -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart -o $@
