# 5208 Project 3 — Graph Ranking HPC

Serial, MPI, OpenMP, and CUDA implementations for graph ranking and parallel numerical experiments on Vanda.

| Part | Start here | Scope |
| --- | --- | --- |
| A | [Part A guide](part-a/README.md) · [Source and results](part-a/) | Serial and MPI PageRank, correctness checks, and CPU scaling |
| B | [Part B guide](#part-b--hybrid-mpiopenmp-pagerank) · [ZIP download](graph-ranking-hpc-b-final.zip) | Hybrid MPI–OpenMP PageRank, graph structure, and scheduling |
| C | [Part C guide](part-c/README.md) · [Source and results](part-c/) | CUDA PageRank, MPI–OpenMP eigenvalue experiments, and CPU/GPU Lanczos |

Run commands from the corresponding part's directory. For Part B, download and extract the ZIP first. The original [Part A ZIP](graph-ranking-hpc-a-final.zip) and [Part C ZIP](graph-ranking-hpc-c-final.zip) are also available.

## Part B — Hybrid MPI–OpenMP PageRank

Part B contains Member B's hybrid PageRank implementation, Vanda benchmark records, analysis scripts, and figures for the graph-structure experiments.

### What changed from the serial/MPI baseline

The serial (V0) and pure-MPI (V1) implementations use incoming-edge CSR, damping factor 0.85, explicit dangling-mass redistribution, and synchronous score updates. The hybrid executable in `src/pagerank_hybrid.cpp` retains the numerical update and MPI collectives and parallelizes local work with OpenMP.

- **V2 (vertex-balanced)** assigns each MPI rank a contiguous destination-vertex range with approximately the same number of vertices.
- **V3 (edge-balanced)** uses incoming-edge prefix sums to choose contiguous ranges with approximately the same number of edges per rank.
- **Scheduling** can be static or dynamic. The dynamic experiments use a chunk size of 64. MPI ranks and OpenMP threads per rank can vary at a fixed nominal CPU budget.

Each output score is written by one thread, avoiding atomic accumulation. Each rank retains the complete score vector. Both variants use two scalar `MPI_Allreduce` calls and one `MPI_Allgatherv` per iteration.

### Completed experiments

| Batch | Vanda job | Runs | Graphs and purpose |
| --- | --- | ---: | --- |
| B1 | `1515568` | 35 | OpenMP scaling on a 100K-vertex / 800K-edge uniform graph with 1, 2, 4, 8, and 12 threads and a scheduling control. |
| B2 | `1515643` | 50 | Pure MPI versus 6×2, 3×4, and 2×6 MPI × OpenMP layouts at a nominal 12-core budget, with partitioning and scheduling controls. |
| B3 | `1515702` | 80 | Uniform, hub, ordered-community, and shuffled-community graphs at 500K vertices / 4M edges, comparing static/dynamic scheduling and V2/V3 at 6×2. |
| B4 | `1515718` | 180 | Uniform and hub graphs at 1M vertices / 8M edges, comparing V2/V3 across three seeds and three process/thread layouts. |
| **Total** | | **345** | Five timing repetitions per configuration. |

The official B1–B4 analysis uses the four job directories above. Other directories under `results/` contain preliminary runs, diagnostics, or separate baselines.

### Main findings

- **OpenMP.** V2 took 0.020979 s at 12 threads and 0.134241 s at one thread, giving **6.40×** speedup and **53.3%** efficiency on the B1 graph.
- **Hybrid layout.** On the B2 uniform graph, V2 at 2 MPI ranks × 6 threads took 0.021812 s versus 0.029782 s for 12-rank pure MPI, a **1.37×** speedup. V2 at 6×2 was slightly slower than pure MPI.
- **Hub partitioning.** On the B3 hub graph, V3 static took 0.227092 s versus 0.631156 s for V2 static, a **2.78×** iteration-loop speedup. Edge-count imbalance dropped from approximately 5.0 to 1.0. The program-internal end-to-end speedup was **1.51×**, excluding external MPI launch and PBS overhead.
- **Robustness.** On B4 hub graphs, the median paired V2/V3 speedups across three seeds were **2.75×** at 6×2, **2.53×** at 3×4, and **2.00×** at 2×6. Uniform graphs showed little or no benefit from edge balancing.
- **Scheduling and ordering.** Dynamic OpenMP scheduling helped the unbalanced V2 hub case. Ordered and shuffled community graphs shared the same topology and had different runtimes. Cache locality is a possible explanation and was not measured directly.

Iteration-loop timings cover 50 PageRank iterations and exclude graph generation and preprocessing. Results are five-run medians. For B4, ratios pair configurations within each graph seed and layout before summarizing across seeds.

### Directory guide

| Path | Contents |
| --- | --- |
| `src/` | Hybrid implementation, graph generator, graph statistics, and shared graph utilities. |
| `baseline_a/` | Copies of the serial and pure-MPI reference sources used for the B handoff. |
| `scripts/` | PBS scripts, experiment configurations, benchmark runner, summarization, and validation utilities. |
| `results/` | Run-level CSVs, command metadata, stdout/stderr, and other Vanda evidence. |
| `analysis/plot_results.py` | Reproduces the official B1–B4 summary tables and twelve figures from archived results. |
| `tables/` | Aggregated CSVs, paired seed comparisons, and data-quality checks. |
| `figures/` | Twelve numbered figures in PNG and vector PDF. |
| `report/` | Part B result notes and figure guide. |
| `Makefile` | Builds the CPU executables with MPI/OpenMP compilers. |

### Reproduce the analysis and figures

From the extracted `graph_ranking_part_b` directory:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/plot_results.py
```

The analysis uses Python, NumPy, pandas, and Matplotlib and reads the archived CSVs. It can run locally without MPI, Vanda access, or regeneration of large graphs. A verified run prints:

```text
Validated 345 official records; generated 12 PNG + 12 PDF figures and summary tables.
```

The package retains the graph generator, parameters, seeds, and metadata needed to regenerate inputs. To rebuild the executables, run `make` with MPI/OpenMP available. Rerunning the Vanda benchmarks also requires a PBS allocation and the regenerated graph edge lists.

### Integrate with the group report

- Match the report's B1–B4 results to the four official job IDs. Compare configurations within the same graph, layout, and timing scope. B1/B2 and B3/B4 ran on different Vanda hosts.
- Match the report's LaTeX `\includegraphics` paths to the numbered files in `figures/`, such as `11_hub_bottleneck.png`.
- `compute_s`, `allreduce_s`, and `allgatherv_s` are separately maximized over MPI ranks. Use `iteration_s` for total loop time. Time inside collectives can include synchronization waits.
- B4 uses three independent graph seeds and five timing repeats per configuration. The range across seeds is descriptive.
- Use Part A for multi-node MPI strong and weak scaling and Part C for CUDA and Lanczos. Large-graph CPU/GPU full-vector comparisons rely on their separately archived validation evidence.

### Provenance and scope

The package was assembled from Vanda outputs backed up on 2026-11-04. The hybrid experiments use synthetic graphs and single-node allocations with a nominal 12-core budget. The results characterize those configurations. Cache misses, network bandwidth, and power consumption were not measured directly.
