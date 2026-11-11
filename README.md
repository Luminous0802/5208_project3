# 5208_project3
MPI, OpenMP and CUDA graph-ranking project

# 5208 Project 3 — Graph Ranking HPC

This repository contains two implementation packages for graph ranking and parallel numerical experiments on Vanda.

| Part | Archive | Scope |
| --- | --- | --- |
| [A](#part-a--serial-and-mpi-pagerank) | [graph-ranking-hpc-a-final.zip](graph-ranking-hpc-a-final.zip) | Serial and MPI PageRank, correctness checks, and CPU scaling |
| [C](#part-c--gpu-pagerank-and-eigenvalue-experiments) | [graph-ranking-hpc-c-final.zip](graph-ranking-hpc-c-final.zip) | CUDA PageRank, MPI+OpenMP eigenvalue experiments, and CPU/GPU Lanczos |

Download and extract the archive for the part you want to run. Run each command from the corresponding extracted project directory. Paths shown below refer to files inside those archives.

## Part A — Serial and MPI PageRank

This folder contains Member A's C++ PageRank work: a serial implementation (V0), an MPI implementation (V1), graph generation, correctness checks, Vanda PBS scripts, and archived measurements.

### Contents

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

### Graph format and algorithm

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

### Build

On a Vanda login node, run these commands from the project directory:

```bash
module load openmpi/gcc/64/4.1.8

g++ -std=c++11 -O2 -Wall -Wextra main.cpp -o pagerank
mpicxx -std=c++11 -O2 -Wall -Wextra pagerank_mpi.cpp -o pagerank_mpi
g++ -std=c++11 -O2 -Wall -Wextra make_pilot_graph.cpp -o make_pilot_graph
g++ -std=c++11 -O2 -Wall -Wextra validate_reference.cpp -o validate_reference
```

The serial program, graph generator, and reference program can also be built with a local C++ compiler. MPI compilation requires an MPI C++ compiler.

### Correctness checks

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

#### Complete-vector validation

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

### Recreate benchmark graphs

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

### Vanda PBS scripts

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

### Archived results and timing

Every performance configuration uses 50 fixed PageRank iterations and five separate program runs. The report uses the median `iteration_s`; `experiment_csv/` retains all individual results. Files named `serial_scaling_*.csv`, `serial_weak_baselines_*.csv`, `mpi_scaling_*.csv`, `mpi_sparsity_*.csv`, and `mpi_weak_*.csv` contain the corresponding experiments. Job IDs in the filenames identify separate measurement batches.

`iteration_s` covers the iteration loop. It excludes graph generation, input reading, validation, edge sorting, CSR construction, and output. V1 reports the maximum loop time across MPI processes. Its `compute_s`, `allreduce_s`, and `allgatherv_s` fields are separate maxima across processes; their sum need not equal `iteration_s`. `edge_imbalance` is the maximum local incoming-edge count divided by the mean, `E/P`.

`rank_sum` checks score conservation. `residual` is the L1 change between the final two score vectors. These values are printed to ten decimal places in ordinary program output, so a displayed residual of `0.0000000000` is subject to rounding. The CSV files record MPI process counts. PBS allocation and process-binding records are needed to establish physical-core placement.

### Serial-reference speedup on weak-scaling graphs

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

## Part C — GPU PageRank and Eigenvalue Experiments

This package contains C++/CUDA implementations, Vanda job scripts, recorded results, and local analysis tools for three completed experiments: single-GPU PageRank against same-node CPU baselines (job **1520880**), power iteration against Lanczos on a shared symmetric operator (job **1526107**), and same-node CPU/GPU Lanczos (job **1536096**).

The recorded evidence comprises ten original files: two PageRank GPU CSVs, five CPU eigenvalue CSV/JSON files, and three GPU Lanczos CSVs. Both eigenvalue experiments retain separate **48-row validation CSVs**. No solver or cluster experiment was rerun during this documentation update. See `C_WORK_SUMMARY.md` for contributions, experimental design, observations, and remaining evidence limitations.

Group-wide CPU strong and weak scaling is Member A's responsibility. Duplicate Part C scaling workflows have been removed. `baseline_a/` retains the shared serial/MPI references; the 12-core hybrid configurations used for the GPU comparison remain included.

### 1. Package contents

| Path | Purpose |
|---|---|
| `src/`, `baseline_a/` | CUDA, serial, MPI, hybrid, power-iteration, and Lanczos sources |
| `Makefile` | CPU/MPI/OpenMP and CUDA build targets |
| `gpu_lanczos/` | GPU Lanczos kernels, MPI host driver, graph specification, build and analysis tools |
| `scripts/run_vanda_gpu_lanczos.pbs` | Four-CPU/one-GPU MPICH Lanczos workflow |
| `scripts/audit_gpu_lanczos.py` | Local source-fingerprint and 300-run/48-validation evidence audit |
| `scripts/run_vanda_gpu_full.pbs` | Single-GPU and 12-core CPU experiment |
| `scripts/run_vanda_eigen_mpich.pbs` | MPICH eigenvalue experiment with 4 ranks and 2 threads per rank |
| `scripts/run_vanda_eigen.pbs` | Equivalent MPICH submission entry point |
| `scripts/visualize_vanda.py` | Local plotting from recorded Vanda performance `runs.csv` files |
| `scripts/audit_eigen_validation.py` | Local numerical audit of the recovered validation CSV; no solver execution |
| `scripts/integrate_results.py` | Explicit import of A/B/GPU/eigenvalue summaries into an integration index |
| `results/Vanda_gpu_1520880/` | Original GPU/CPU measurements and summary |
| `results/Vanda_eigen_1526107/` | Original performance records, validation records, and provenance |
| `results/Vanda_gpu_lanczos_1536096/` | Original GPU Lanczos measurements, summary, and validation CSV |
| `configs/part_b_graphs.json` | Graph sizes, seeds, generation options, and input hashes |
| `tests/` | Small graph inputs and MPI diagnostic source |
| `inputs/`, `figures/` | Locations for additional downloaded evidence and locally generated figures |

`CHECKSUMS.json` records package-file hashes. All ten original result files retain their original bytes. Generated binaries, historical local performance measurements, A/B result archives, and pre-generated figures are not included.

### 2. Recorded experiments

#### GPU and 12-core CPU comparison: job 1520880

`results/Vanda_gpu_1520880/runs.csv` contains **210 measurements**: 42 configurations with five repeats each. The device was an NVIDIA A40 on GN-A40-004. All measurements used FP64 and 50 PageRank iterations.

The six graph files match Member B's input hashes. Each graph was tested with a serial CPU reference, four 12-core MPI+OpenMP configurations, and CUDA thread-per-vertex and warp-per-vertex mappings. The original `summary.csv` is retained. `IMPORT_RECORD.json` records file, summary-statistic, baseline, graph, and source-hash checks; it is not a new measurement.

Only the two supplied GPU CSV files are included. Original GPU preflight JSON, device metadata, PBS allocation records, and binding logs were not supplied.

#### Power iteration and Lanczos: job 1526107

Under `results/Vanda_eigen_1526107/`:

| File | Contents |
|---|---|
| `comparison/runs.csv` | 165 performance measurements: 33 configurations, five repeats each |
| `comparison/summary.csv` | Original 33-row Vanda performance summary |
| `comparison/provenance.json` | Job/node identity, MPI layout, hashes, parameters, and timing definitions |
| `validation/summary.json` | Original overall PASS for 48 small-graph solver evaluations |
| `validation/comparisons.csv` | Recovered 48-row validation results, including Ritz values and independent dense residuals |
| `IMPORT_RECORD.json` | Historical packaging checks for the first four original files |
| `validation/IMPORT_RECORD.json` | Import identity and hash checks for the recovered validation CSV |

The recorded site is Vanda, job `1526107.stdct-mgmt-02`, node IN-001, MPICH 4.1.1, with 4 MPI ranks and 2 OpenMP threads per rank. Recorded C++ source hashes and the `scripts/compare_eigen.py` hash match the supplied code. Validation and performance sampling have different `measurement_id` values.

Both methods solve the same lazy symmetric eigenproblem `B=(I+S)/2`. This is a separate problem from directed PageRank with damping 0.85. `solve_s` covers numerical solving, projected solving/reconstruction where applicable, and direct residual evaluation; it excludes graph loading and MPI startup. Fixed-budget results do not establish the exact minimum time to a tolerance or a serial-to-parallel speedup.

#### CPU/GPU Lanczos: job 1536096

The new `gpu_lanczos/` module contains handwritten CUDA kernels for sparse products, vector operations, two-pass full reorthogonalization, and Ritz-vector reconstruction. MPI communicates through host buffers; the projected eigenproblem is solved on the host. The shared CPU sources and helpers match the existing project byte for byte. The module's complete 15-file source fingerprint matches the recorded GPU Lanczos timing CSV.

`results/Vanda_gpu_lanczos_1536096/` contains 300 measurements, the original 60-row summary, and `validation/comparisons.csv` with 48 records. The experiment used one NVIDIA A40 on GN-A40-003, three N=100000/input-E=800000 graphs, FP64, the lazy operator, budgets 5/10/20/30/50, four variants, and five repeats. The variants are serial CPU, 2x2 MPI+OpenMP CPU, one-process GPU, and two-process shared-GPU execution. Two GPU processes share one physical device.

At common passing budgets 20/20/30, GPU-1 solve ratios over the same-job 2x2 CPU baseline are 4.33/3.08/4.66 for Uniform/Hub/Community. Broader application ratios are 0.87--1.07. These are separate from the earlier Power/Lanczos method comparison and are not multi-GPU scaling results.

Audit existing evidence locally:

```bash
python3 scripts/audit_gpu_lanczos.py
```

For a new Vanda run, submit from the project root:

```bash
qsub scripts/run_vanda_gpu_lanczos.pbs
```

The workflow requests four CPUs, two MPI slots, and one GPU, loads MPICH/CUDA, builds into an isolated directory, validates, and then benchmarks. The complete design, thresholds, timing definitions, pilot option, and plotting commands are in `gpu_lanczos/README.md`. Existing results need no rerun for this integration.

### 3. What the eigenvalue validation checks

The validation CSV covers six graphs: a zero-edge graph, a single vertex, a two-vertex edge, a path, a cycle, and a self-loop/isolated-vertex case. Each is evaluated with power iteration and Lanczos, serial execution and four-rank MPI execution, and one-step and larger budgets.

An independent dense construction of `B` supplies the residual `||Bq - lambda*q||_2`. For endpoint accuracy, the archived validator uses both eigenvalue error and residual below `1e-8`. Endpoint rows have budget greater than one, or `N=1`; generic one-step rows test consistency and are not required to converge. Across the selected endpoints, the maximum dense residual is approximately **1.0180e-10**. Serial power/Lanczos one-step comparisons have maximum recorded eigenvalue difference 0 and maximum residual difference approximately **1.1102e-16**, below the archived `1e-12` criterion.

Run the local audit from the project root:

```bash
python3 scripts/audit_eigen_validation.py
```

It checks all 48 row identities against the original summary, verifies the case/configuration coverage and recorded source hashes, and prints the six endpoint summaries and six serial one-step comparisons. It reads existing evidence without compiling, launching MPI, requesting a GPU, or writing new measurements. The thresholds are criteria in the archived validator; its driver hash was not recorded in the original Vanda summary.

The CSV does not contain vector dumps or the solver-reported residuals needed to reconstruct vector-norm errors or dense-versus-reported residual differences. Those checks retain the original suite PASS evidence only. Detailed methods and the six-case numerical table are in `docs/VANDA_EIGEN.md`.

### 4. Generate figures locally

Install plotting dependencies from the project root:

```bash
python3 -m pip install -r requirements-visualization.txt
```

Plot the recorded eigenvalue performance measurements:

```bash
python3 scripts/visualize_vanda.py eigen \
  --runs results/Vanda_eigen_1526107/comparison/runs.csv \
  --out figures/Vanda_eigen_1526107
```

Plot the recorded GPU/CPU measurements:

```bash
python3 scripts/visualize_vanda.py gpu \
  --runs results/Vanda_gpu_1520880/runs.csv \
  --out figures/Vanda_gpu_1520880
```

Plot the recorded GPU Lanczos measurements with their separate module tool:

```bash
python3 gpu_lanczos/scripts/analyze_gpu_lanczos.py \
  results/Vanda_gpu_lanczos_1536096/runs.csv \
  --out figures/Vanda_gpu_lanczos_1536096/summary.csv \
  --plot-dir figures/Vanda_gpu_lanczos_1536096/plots
```

`--runs` takes a performance `runs.csv`, not `summary.csv` or validation `comparisons.csv`. These commands analyze existing records and do not run C++ solvers or submit jobs. Outputs include a regenerated `summary.csv`, PNG/PDF figures under `plots/`, and `plot_provenance.json`. Original inputs remain unchanged. Use a fresh empty `--out` directory for each plotting run.

Error bars show observed min/max, not confidence intervals. `iteration_s` is host-observed loop time; `gpu_timeline_s` is a CUDA-event interval. H2D + loop + D2H excludes allocation and warmup and is not full end-to-end time. The existing GPU plotter uses linear axes; changing the plot scale requires no new measurements. Eigenvalue figures use logarithmic residual/time axes and floor displayed zero residuals at `1e-17` without changing the CSV.

### 5. Integrate A, B, and C summaries

Pass the actual summaries explicitly, for example:

```bash
python3 scripts/integrate_results.py \
  --gpu-summary results/Vanda_gpu_1520880/summary.csv \
  --eigen-summary results/Vanda_eigen_1526107/comparison/summary.csv \
  --out results/my_vanda_integration
```

Member A and B summaries can be passed through `--part-a-summary` and `--part-b-summary`. `--cpu-summary` is a compatibility alias for Member A's input; it does not refer to a duplicate Part C scaling experiment. Inputs must retain their actual Vanda site/job fields. A/B performance data were not supplied with this package.

The new GPU Lanczos summary is analyzed with its own module tool; the original integration command does not automatically import it. The integration tool creates an index and combined tables. It does not write the final report, generate A/B figures, or compute cross-hardware/cross-algorithm speedups. Check graph identity, iteration rules, precision, hardware, and timing scope before combining comparisons.

### 6. Build and reproduce on Vanda

With a C++17 compiler, MPI, and OpenMP available:

```bash
make cpu CXX=g++ MPICXX=mpicxx
```

With the CUDA Toolkit available, the measured A40 target can be built with:

```bash
make gpu CUDA_ARCH=sm_86
```

For another GPU, select its supported architecture. Use the same MPI implementation for compilation and execution. The Vanda scripts load the required modules and use isolated build directories. See `docs/VANDA_RUN.md` for submission instructions and `docs/VANDA_EIGEN.md` for eigenvalue details.

The optional GPU Lanczos build has its own target and output subdirectory:

```bash
make gpu-lanczos CXX=g++ MPICXX=mpicxx CUDA_ARCH=sm_86
```

The module accepts `CUDA_HOME` and `CUDA_LIBDIR`; the Vanda PBS script selects the actual toolkit paths. Its default binaries are isolated from the PageRank build.

The project implements its numerical kernels and projected solver directly; it does not use BLAS/LAPACK solver libraries. Existing results can be analyzed without rebuilding or rerunning completed experiments.

