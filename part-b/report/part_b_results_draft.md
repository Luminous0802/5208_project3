# Part B — Hybrid MPI–OpenMP PageRank: Experimental Results and Discussion

## Scope and implementation

We extend the existing serial (V0) and pure MPI (V1) PageRank baselines with hybrid MPI–OpenMP execution. V2 partitions destination vertices evenly across MPI ranks, whereas V3 selects contiguous destination-vertex ranges to approximately balance incoming-edge counts. Both use OpenMP for local vertex updates and support static and dynamic loop scheduling. The PageRank damping factor is 0.85, with dangling-node mass redistributed uniformly. The graph is represented using incoming-edge CSR, and MPI collectives synchronize global state each iteration. The core numerical loop is implemented in C++ with MPI/OpenMP rather than delegated to BLAS/LAPACK.

## Experimental protocol and validation

Experiments were run on NUS Vanda CPU nodes. The reported `iteration_s` measures **50 fixed PageRank iterations**, excluding graph loading and partition setup. Each configuration was repeated five times in randomized order; the median is reported. MPI process mapping was explicitly configured with `--map-by slot:PE={threads} --bind-to core`, with OpenMP thread binding enabled. Initial runs without explicit process/thread binding were excluded. The official datasets contain B1=35, B2=50, B3=80 and B4=180 observations, totaling 345. All recorded runs completed 50 iterations and have finite positive iteration times and PageRank rank sums within 1e-8 of unity. These checks supplement, but do not replace, small-graph reference-output validation. The rank-sum test alone cannot establish vector-level accuracy.

## B1 — OpenMP thread scaling

On a uniform random directed graph (N=100,000, E=800,000), the V2 12-thread configuration achieved approximately **6.40x speedup** over the V2 single-thread configuration, corresponding to about **53.3% parallel efficiency**. Speedup was sublinear, consistent with overheads from synchronization, memory access, and thread management. V0 and V2 single-thread runtimes should not be conflated when defining within-implementation speedup.

## B2 — Fixed-core hybrid comparison

With 12 allocated CPU cores on the same 100K/800K uniform graph, pure MPI V1 (12 ranks x 1 thread) had median iteration time **0.029782 s**. V2 (6x2) required **0.031544 s**, V2 (3x4) **0.024232 s**, and V2 (2x6) **0.021812 s**. Thus V2 (2x6) was approximately **1.37x faster than pure MPI V1**, whereas V2 (6x2) was slower. Median V1 Allgatherv maximum was **0.016847 s**, compared with **0.006447 s** for V2 (2x6); this supports the observation that fewer ranks can reduce collective overhead, but does not by itself establish causation. The B2 phase-maxima figure compares compute, Allreduce and Allgatherv independently; maxima may come from different ranks and **must not be summed** to reconstruct iteration time.

## B3 — Graph structure, partitioning and scheduling

Holding N=500,000, E=4,000,000 and 6 MPI ranks x 2 OpenMP threads constant, the Hub graph produced a striking partitioning effect. Vertex-balanced V2 with static scheduling took **0.631156 s** (median) and had edge imbalance **4.9992**, while edge-balanced V3 took **0.227092 s** with imbalance **1.0004**. This is a **2.78x speedup** and about **64.0% lower iteration time**. The per-rank maximum compute time fell from **0.576534 s** to **0.138233 s**, consistent with better distributed edge workloads.

Dynamic scheduling reduced V2 Hub time from 0.631156 s to 0.359222 s, but V3 static was faster at 0.227092 s. OpenMP dynamic scheduling redistributes iterations **within each rank**; it does not move excess edges from an overloaded MPI rank to another rank. On Uniform and Community graphs, V2 and V3 had edge imbalance near one and similar performance, so additional edge balancing offered little benefit.

For the Community graph, V3 static took **0.102836 s** with ordered vertex IDs versus **0.124680 s** after relabeling, a **21.2% higher runtime** for shuffled IDs. The graph generator creates topology first and then applies an independently seeded vertex permutation; thus the two graphs are isomorphic, but memory locality is a plausible rather than directly measured explanation.

## B4 — Scale and random-seed robustness

On larger N=1,000,000, E=8,000,000 Uniform and Hub graphs, three seeds (20261029–20261031) were used. For Hub graphs, the median V2/V3 speedup across seeds was **2.75x (6x2)**, **2.52x (3x4)**, and **2.00x (2x6)**. Corresponding V2 edge imbalance was approximately 5.0, 2.6 and 1.8, whereas V3 remained approximately 1.0. For Uniform graphs, paired speedups remained near 1.0. This reproduces the qualitative structure-dependent result at a second graph scale and across multiple graph realizations. The three seeds are independent graph realizations, not independent compute platforms.

## Limitations and interpretation

1. Only two main graph sizes were used for the structure comparison; general scaling laws should not be inferred.
2. Five repeats provide medians and observed ranges, not a guarantee of statistical significance; tiny differences on Uniform graphs should not be overstated.
3. Reported `compute_s`, `allreduce_s`, and `allgatherv_s` are separate maxima over ranks; do not stack or sum them. MPI time includes synchronization and waiting, not solely network transfer.
4. The timing comparison focuses on iteration time; load and partition time are reported separately and may matter for end-to-end use.
5. The B4 graph edge lists were stored in scratch and are not included in the backup. The generator source, parameters and seeds enable reconstruction.
6. Pure MPI strong/weak scaling to at least 256 cores and the required GPU test are separate whole-team deliverables; the Part B results do not substitute for them.

## Recommended report figures

- `figure_b1_speedup.png`: OpenMP thread speedup vs ideal linear speedup.
- `figure_b2_hybrid.png`: Pure MPI vs hybrid configurations at fixed 12 cores.
- `figure_b2_phase_maxima.png`: MPI/compute phase maximum timings (not stacked).
- `figure_b3_structure.png`: Graph-structure sensitivity of V2 vs V3.
- `figure_b3_hub_phases.png`: Hub iteration/compute/Allgatherv comparisons.
- `figure_b4_robustness.png`: Paired V2/V3 speedup across seeds.
- `figure_b4_imbalance.png`: Partition imbalance by graph and rank count.

## Reproducibility

Use the four official results folders `b1_openmp_1515568`, `b2_hybrid_1515643`, `b3_graph_1515702`, and `b4_large_1515718`. Exclude incomplete or diagnostic runs. Preserve `runs.csv`, `metadata.json`, the generator and solver source, PBS scripts, and configuration JSON. All speedup comparisons should be matched on graph, seed, core layout and timing scope.
