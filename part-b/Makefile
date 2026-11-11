CXX = c++
MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -Wpedantic
OMPFLAGS = -fopenmp
LDLIBS =
BIN = bin

.PHONY: all clean test
all: $(BIN)/pagerank $(BIN)/pagerank_mpi $(BIN)/pagerank_hybrid $(BIN)/make_graph $(BIN)/graph_stats
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
test: all
	python3 scripts/validate.py
clean:
	rm -rf bin
