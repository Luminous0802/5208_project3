# Graph Ranking — Part B

Hybrid MPI–OpenMP PageRank: implementation, performance experiments, and analysis.

## Contents

Browse the [figures](figures/), [experiment results](results/), and [report notes](report/).

- `src/`: C++ implementation (V2/V3), graph generator and graph statistics.
- `baseline_a/`: A's serial and pure-MPI reference implementations.
- `scripts/`: original Vanda PBS jobs, benchmark runner and experiment configuration files.
- `results/`: run-level benchmark CSVs for B1 `1515568`, B2 `1515643`, B3 `1515702`, and B4 `1515718`. These are the inputs used to generate the figures.
- `analysis/plot_results.py`: full reproducible validation, aggregation and plotting code.
- `tables/`: generated configuration summaries, data checks and paired speedup tables.
- `figures/`: 12 publication-style figures in PNG (260 dpi) and vector PDF formats.
- `report/`: report draft and figure guide.

## Reproduce the analysis

```bash
cd part-b
python3 -m pip install -r requirements.txt
python3 analysis/plot_results.py
```

No MPI, Vanda access, or graph reconstruction is needed for plotting. The script asserts 345 valid official measurements, five repeats per configuration, fixed 50 iterations, and rank-sum tolerance. It does not claim to validate full PageRank vectors on all large graphs.

## What changed from the serial/MPI baseline

The serial (V0) and pure-MPI (V1) implementations use incoming-edge CSR, damping factor 0.85, explicit dangling-mass redistribution, and synchronous score updates. The hybrid executable in `src/pagerank_hybrid.cpp` retains the numerical update and MPI collectives and parallelizes local work with OpenMP.

- **V2 (vertex-balanced)** assigns each MPI rank a contiguous destination-vertex range with approximately the same number of vertices.
- **V3 (edge-balanced)** uses incoming-edge prefix sums to choose contiguous ranges with approximately the same number of edges per rank.
- **Scheduling** can be static or dynamic. The dynamic experiments use a chunk size of 64. MPI ranks and OpenMP threads per rank can vary at a fixed nominal CPU budget.

Each output score is written by one thread, avoiding atomic accumulation. Each rank retains the complete score vector. Both variants use two scalar `MPI_Allreduce` calls and one `MPI_Allgatherv` per iteration.

## Completed experiments

| Batch | Vanda job | Runs | Graphs and purpose |
| --- | --- | ---: | --- |
| B1 | `1515568` | 35 | OpenMP scaling on a 100K-vertex / 800K-edge uniform graph with 1, 2, 4, 8, and 12 threads and a scheduling control. |
| B2 | `1515643` | 50 | Pure MPI versus 6×2, 3×4, and 2×6 MPI × OpenMP layouts at a nominal 12-core budget, with partitioning and scheduling controls. |
| B3 | `1515702` | 80 | Uniform, hub, ordered-community, and shuffled-community graphs at 500K vertices / 4M edges, comparing static/dynamic scheduling and V2/V3 at 6×2. |
| B4 | `1515718` | 180 | Uniform and hub graphs at 1M vertices / 8M edges, comparing V2/V3 across three seeds and three process/thread layouts. |
| **Total** | | **345** | Five timing repetitions per configuration. |

The official B1–B4 analysis uses the four job directories above. Other directories under `results/` contain preliminary runs, diagnostics, or separate baselines.

## Main findings

- **OpenMP.** V2 took 0.020979 s at 12 threads and 0.134241 s at one thread, giving **6.40×** speedup and **53.3%** efficiency on the B1 graph.
- **Hybrid layout.** On the B2 uniform graph, V2 at 2 MPI ranks × 6 threads took 0.021812 s versus 0.029782 s for 12-rank pure MPI, a **1.37×** speedup. V2 at 6×2 was slightly slower than pure MPI.
- **Hub partitioning.** On the B3 hub graph, V3 static took 0.227092 s versus 0.631156 s for V2 static, a **2.78×** iteration-loop speedup. Edge-count imbalance dropped from approximately 5.0 to 1.0. The program-internal end-to-end speedup was **1.51×**, excluding external MPI launch and PBS overhead.
- **Robustness.** On B4 hub graphs, the median paired V2/V3 speedups across three seeds were **2.75×** at 6×2, **2.53×** at 3×4, and **2.00×** at 2×6. Uniform graphs showed little or no benefit from edge balancing.
- **Scheduling and ordering.** Dynamic OpenMP scheduling helped the unbalanced V2 hub case. Ordered and shuffled community graphs shared the same topology and had different runtimes. Cache locality is a possible explanation and was not measured directly.

Iteration-loop timings cover 50 PageRank iterations and exclude graph generation and preprocessing. Results are five-run medians. For B4, ratios pair configurations within each graph seed and layout before summarizing across seeds.

## Experimental interpretation

- Time values represent 50 PageRank iterations unless explicitly labeled `end-to-end`.
- `compute_s`, `allreduce_s`, and `allgatherv_s` are maxima over MPI ranks and **must not be summed or stacked**.
- B4 speedup pairs V2 and V3 **within the same graph seed and MPI/OpenMP layout**; the range across three seeds is descriptive, not a confidence interval.
- A's MPI sparsity analysis is separate. Pure-MPI 256-core scaling and GPU tests are whole-team requirements, not covered by this Part B archive.
- B4 generated graph edge lists were kept on Vanda scratch and are not included. Generator source, graph parameters, seeds and output hashes remain available.

## Source and benchmark provenance

Vanda compiled binaries, original CSV logs and metadata were backed up on 2026-11-04. This directory contains the four official runs used for the figures. Raw scheduler logs and environment captures are excluded.
