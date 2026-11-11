# Graph Ranking HPC

This folder contains Member A's C++ PageRank work: a serial implementation (V0), an MPI implementation (V1), graph generation, correctness checks, Vanda PBS scripts, and archived measurements. The group's MPI+OpenMP, graph-structure, GPU, and Lanczos files can be added when those contributions are integrated.

## Contents

| File or directory | Purpose |
| --- | --- |
| `main.cpp` | Serial PageRank using incoming-edge CSR storage. |
| `pagerank_mpi.cpp` | MPI PageRank with contiguous destination-vertex partitions. |
| `validate_reference.cpp` | Independent edge-scatter C++ reference for complete-vector validation. |
| `make_pilot_graph.cpp` | Seeded generator for directed synthetic graphs. |
| `tiny_graph.txt`, `check_graph.txt`, `duplicate_graph.txt` | Small correctness and input-validation cases. |
| `validation_graph_10k_80k.txt` | Fixed 10,000-vertex graph for complete-vector validation. |
| `validation_rank_*.txt` | Saved V0, V1, and reference score vectors for that graph. |
| `graph_hashes.sha256` | SHA-256 hashes of two baseline pilot graphs. |
| `scripts/` | PBS scripts used on Vanda. |
| `experiment_csv/` | Individual measurements from completed experiments. |

Large pilot and weak-scaling graphs are excluded from this folder to keep the GitHub upload small. The graph generator and parameters below recreate the inputs. The archived CSV measurements remain available without the graph files.

## Graph format and algorithm

A graph file starts with `N E`, followed by exactly `E` lines of `source destination`. Vertex identifiers range from `0` to `N-1`. For example:

```text
2 1
0 1
```

V0 and V1 use damping factor `0.85`, a uniform initial score vector, and redistribution of dangling-vertex score. They accept self-loops and reject duplicate directed edges. The generator excludes both self-loops and duplicate edges.

Both PageRank programs accept:

```text
PROGRAM GRAPH_FILE K [TOL] [--dump FILE]
```

`K` is a positive maximum iteration count. Without `TOL`, the program executes exactly `K` iterations. With a positive `TOL`, it stops early when the L1 difference between successive score vectors falls below the threshold. `--dump FILE` writes all final scores, one per line, with 17-digit precision. Scores for graphs of at most 20 vertices are also printed to standard output.

## Build

On a Vanda login node, run these commands from the project directory:

```bash
cd "$HOME/graph-ranking-hpc"
module load openmpi/gcc/64/4.1.8

g++ -std=c++11 -O2 -Wall -Wextra main.cpp -o pagerank
mpicxx -std=c++11 -O2 -Wall -Wextra pagerank_mpi.cpp -o pagerank_mpi
g++ -std=c++11 -O2 -Wall -Wextra make_pilot_graph.cpp -o make_pilot_graph
g++ -std=c++11 -O2 -Wall -Wextra validate_reference.cpp -o validate_reference
```

The serial program, graph generator, and reference program can also be built with a local C++ compiler. MPI compilation requires an MPI C++ compiler.

## Correctness checks

Run the small serial cases:

```bash
./pagerank tiny_graph.txt 1
./pagerank tiny_graph.txt 50 1e-8
./pagerank check_graph.txt 1
./pagerank duplicate_graph.txt 1
```

The first update on `tiny_graph.txt` gives `(0.2875000000, 0.7125000000)`. With tolerance `1e-8`, the recorded V0 and V1 runs stopped after 22 iterations and reported scores near `(0.3508771940, 0.6491228060)`. One update on `check_graph.txt` gives `(0.0906250000, 0.1968750000, 0.6218750000, 0.0906250000)`. The duplicate-edge case should print `Duplicate edge` and exit with a nonzero status.

For an MPI check on Vanda, submit from the project root:

```bash
qsub scripts/vanda_mpi_smoke.pbs
```

### Complete-vector validation

The fixed graph has `N=10000`, `E=80000`, seed `20261110`, and SHA-256 hash `f1320185cb96869d0d3e0ecb53d87f1125e97b9d83f8ab6f9cc6cf406e9a5312`. Run all implementations for 50 fixed iterations:

```bash
./pagerank validation_graph_10k_80k.txt 50 --dump validation_rank_v0.txt
./validate_reference validation_graph_10k_80k.txt 50 validation_rank_ref.txt
qsub scripts/vanda_mpi_validate_10k.pbs
```

The PBS script verifies the graph hash, uses 12 MPI processes, and writes `validation_rank_v1_p12.txt`. Each saved vector contains 10,000 scores. The recorded comparisons were:

| Comparison | L1 difference | Maximum absolute difference |
| --- | ---: | ---: |
| V0 versus V1 | `7.1638658546979705e-17` | `5.4210108624275222e-20` |
| V0 versus C++ reference | `1.2974850686041373e-16` | `8.1315162936412833e-20` |
| V1 versus C++ reference | `1.3483748080751756e-16` | `1.0842021724855044e-19` |

The reference traverses an edge list and scatters each contribution to its destination. V0 and V1 traverse incoming-edge CSR rows. The comparison checks every vertex score on this fixed graph.

## Recreate benchmark graphs

The generator accepts `./make_pilot_graph N E SEED OUTPUT_FILE`. The five baseline configurations use seed `20261029`:

```bash
./make_pilot_graph 100000 800000 20261029 pilot_graph.txt
./make_pilot_graph 500000 4000000 20261029 pilot_graph_500k_4m.txt
./make_pilot_graph 1000000 4000000 20261029 pilot_graph_1m_4m.txt
./make_pilot_graph 1000000 8000000 20261029 pilot_graph_1m.txt
./make_pilot_graph 1000000 16000000 20261029 pilot_graph_1m_16m.txt
```

The first, second, and fourth configurations have mean out-degree 8. The graphs with `N=1000000` and 4, 8, or 16 million edges vary edge count at fixed vertex count. The generator uses `std::mt19937_64` and continues drawing edges until it has written the requested number of distinct directed edges.

For exact reproduction of the two original pilot inputs, compare the regenerated files with `graph_hashes.sha256`. The C++ standard does not require `std::uniform_int_distribution` to produce identical samples across every standard-library implementation. If a hash differs, use an archived original graph for an exact-input rerun. The Vanda runs used the original graph files referenced in the PBS scripts.

The weak-scaling script generates its own graph for each process count `P` using `N=5000P`, `E=40000P`, and seed `20261029`.

## Vanda PBS scripts

Submit jobs from the project root. The scripts use `PBS_O_WORKDIR` to find programs and graph inputs, and they load Open MPI 4.1.8 within the job.

| Script | Purpose |
| --- | --- |
| `scripts/vanda_serial_scaling.pbs` | Five serial graph configurations, five runs each; generates missing graphs. |
| `scripts/vanda_mpi_repeat.pbs` | Strong-scaling measurements on `N=1000000`, `E=8000000`. |
| `scripts/vanda_mpi_sparsity.pbs` | Twelve-process comparison of the three `N=1000000` edge counts. |
| `scripts/vanda_mpi_weak.pbs` | Weak scaling with graph size proportional to `P`. |
| `scripts/vanda_serial_weak_baselines.pbs` | Five serial runs on each of the six weak-scaling graphs. |
| `scripts/vanda_mpi_smoke.pbs` | Two-process small-graph check. |
| `scripts/vanda_mpi_validate_10k.pbs` | Twelve-process complete-vector validation. |

The MPI repeat script requires `pilot_graph_1m.txt`. The sparsity script requires `pilot_graph_1m_4m.txt`, `pilot_graph_1m.txt`, and `pilot_graph_1m_16m.txt`. Generate or restore those files before submitting either script. The serial size-scaling and MPI weak-scaling scripts can generate their missing graphs. The serial weak-baseline script reads the six graph files retained from the MPI weak-scaling runs and stops if any is missing.

Submit the fixed-resource jobs with:

```bash
qsub scripts/vanda_serial_scaling.pbs
qsub scripts/vanda_mpi_sparsity.pbs
qsub scripts/vanda_serial_weak_baselines.pbs
```

The process-scaling and weak-scaling scripts require a `RANKS` value matching the allocated MPI slots. For example:

```bash
qsub -N pr_mpi_12 -l select=1:ncpus=12:mpiprocs=12:mem=8gb -v RANKS=12 scripts/vanda_mpi_repeat.pbs
qsub -N pr_weak_12 -l select=1:ncpus=12:mpiprocs=12:mem=8gb -v RANKS=12 scripts/vanda_mpi_weak.pbs
```

The archived studies include `P=12, 24, 48, 96, 192, 264`. For other process counts, adjust the PBS allocation and `RANKS` together. The scripts check the allocated MPI slot count before launching PageRank.

## Archived results and timing

Every performance configuration uses 50 fixed PageRank iterations and five separate program runs. The report uses the median `iteration_s`; `experiment_csv/` retains all individual results. Files named `serial_scaling_*.csv`, `serial_weak_baselines_*.csv`, `mpi_scaling_*.csv`, `mpi_sparsity_*.csv`, and `mpi_weak_*.csv` contain the corresponding experiments. Job IDs in the filenames identify separate measurement batches.

`iteration_s` covers the iteration loop. It excludes graph generation, input reading, validation, edge sorting, CSR construction, and output. V1 reports the maximum loop time across MPI processes. Its `compute_s`, `allreduce_s`, and `allgatherv_s` fields are separate maxima across processes; their sum need not equal `iteration_s`. `edge_imbalance` is the maximum local incoming-edge count divided by the mean, `E/P`.

`rank_sum` checks score conservation. `residual` is the L1 change between the final two score vectors. These values are printed to ten decimal places in ordinary program output, so a displayed residual of `0.0000000000` is subject to rounding. The CSV files record MPI process counts. PBS allocation and process-binding records are needed to establish physical-core placement.

## Serial-reference speedup on weak-scaling graphs

The serial weak-baseline job `1541619` ran V0 five times on each graph used in the MPI weak-scaling study. Its raw measurements are in `experiment_csv/serial_weak_baselines_1541619.csv`. For each graph `G_P`, we divide the five-run median V0 `iteration_s` by the five-run median V1 `iteration_s` at process count `P`:

```text
S_V0(P) = median[T_V0(G_P)] / median[T_V1(P, G_P)]
```

| MPI processes `P` | V0 median (s) | V1 median (s) | Speedup over V0 |
| ---: | ---: | ---: | ---: |
| 12 | 0.044433 | 0.015539 | 2.86 |
| 24 | 0.092754 | 0.027743 | 3.34 |
| 48 | 0.275846 | 0.135779 | 2.03 |
| 96 | 0.969937 | 0.202849 | 4.78 |
| 192 | 2.365833 | 0.901053 | 2.63 |
| 264 | 3.411630 | 0.630826 | 5.41 |

All V0 weak-baseline runs used Vanda host `GN-A40-074`. The MPI runs were scheduled separately and used different allocations. Host differences may affect these measured speedups. Weak-scaling efficiency uses the 12-process V1 run as its reference and has a different denominator.
