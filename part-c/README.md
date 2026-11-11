# Graph Ranking Part C: GPU PageRank and MPI+OpenMP Eigenvalue Experiments

This package contains C++/CUDA implementations, Vanda job scripts, recorded results, and local analysis tools for three completed experiments: single-GPU PageRank against same-node CPU baselines (job **1520880**), power iteration against Lanczos on a shared symmetric operator (job **1526107**), and same-node CPU/GPU Lanczos (job **1536096**).

The recorded evidence comprises ten original files: two PageRank GPU CSVs, five CPU eigenvalue CSV/JSON files, and three GPU Lanczos CSVs. Both eigenvalue experiments retain separate **48-row validation CSVs**. No solver or cluster experiment was rerun during this documentation update. See [C_WORK_SUMMARY.md](C_WORK_SUMMARY.md) for contributions, experimental design, observations, and remaining evidence limitations.

Group-wide CPU strong and weak scaling is Member A's responsibility. Duplicate Part C scaling workflows have been removed. `baseline_a/` retains the shared serial/MPI references; the 12-core hybrid configurations used for the GPU comparison remain included.

## 1. Package contents

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

## 2. Recorded experiments

### GPU and 12-core CPU comparison: job 1520880

`results/Vanda_gpu_1520880/runs.csv` contains **210 measurements**: 42 configurations with five repeats each. The device was an NVIDIA A40 on GN-A40-004. All measurements used FP64 and 50 PageRank iterations.

The six graph files match Member B's input hashes. Each graph was tested with a serial CPU reference, four 12-core MPI+OpenMP configurations, and CUDA thread-per-vertex and warp-per-vertex mappings. The original `summary.csv` is retained. `IMPORT_RECORD.json` records file, summary-statistic, baseline, graph, and source-hash checks; it is not a new measurement.

Only the two supplied GPU CSV files are included. Original GPU preflight JSON, device metadata, PBS allocation records, and binding logs were not supplied.

### Power iteration and Lanczos: job 1526107

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

### CPU/GPU Lanczos: job 1536096

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

The workflow requests four CPUs, two MPI slots, and one GPU, loads MPICH/CUDA, builds into an isolated directory, validates, and then benchmarks. The complete design, thresholds, timing definitions, pilot option, and plotting commands are in [gpu_lanczos/README.md](gpu_lanczos/README.md). Existing results need no rerun for this integration.

## 3. What the eigenvalue validation checks

The validation CSV covers six graphs: a zero-edge graph, a single vertex, a two-vertex edge, a path, a cycle, and a self-loop/isolated-vertex case. Each is evaluated with power iteration and Lanczos, serial execution and four-rank MPI execution, and one-step and larger budgets.

An independent dense construction of `B` supplies the residual `||Bq - lambda*q||_2`. For endpoint accuracy, the archived validator uses both eigenvalue error and residual below `1e-8`. Endpoint rows have budget greater than one, or `N=1`; generic one-step rows test consistency and are not required to converge. Across the selected endpoints, the maximum dense residual is approximately **1.0180e-10**. Serial power/Lanczos one-step comparisons have maximum recorded eigenvalue difference 0 and maximum residual difference approximately **1.1102e-16**, below the archived `1e-12` criterion.

Run the local audit from the project root:

```bash
python3 scripts/audit_eigen_validation.py
```

It checks all 48 row identities against the original summary, verifies the case/configuration coverage and recorded source hashes, and prints the six endpoint summaries and six serial one-step comparisons. It reads existing evidence without compiling, launching MPI, requesting a GPU, or writing new measurements. The thresholds are criteria in the archived validator; its driver hash was not recorded in the original Vanda summary.

The CSV does not contain vector dumps or the solver-reported residuals needed to reconstruct vector-norm errors or dense-versus-reported residual differences. Those checks retain the original suite PASS evidence only. Detailed methods and the six-case numerical table are in [docs/VANDA_EIGEN.md](docs/VANDA_EIGEN.md).

## 4. Generate figures locally

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

## 5. Integrate A, B, and C summaries

Pass the actual summaries explicitly, for example:

```bash
python3 scripts/integrate_results.py \
  --gpu-summary results/Vanda_gpu_1520880/summary.csv \
  --eigen-summary results/Vanda_eigen_1526107/comparison/summary.csv \
  --out results/my_vanda_integration
```

Member A and B summaries can be passed through `--part-a-summary` and `--part-b-summary`. `--cpu-summary` is a compatibility alias for Member A's input; it does not refer to a duplicate Part C scaling experiment. Inputs must retain their actual Vanda site/job fields. A/B performance data were not supplied with this package.

The new GPU Lanczos summary is analyzed with its own module tool; the original integration command does not automatically import it. The integration tool creates an index and combined tables. It does not write the final report, generate A/B figures, or compute cross-hardware/cross-algorithm speedups. Check graph identity, iteration rules, precision, hardware, and timing scope before combining comparisons.

## 6. Build and reproduce on Vanda

With a C++17 compiler, MPI, and OpenMP available:

```bash
make cpu CXX=g++ MPICXX=mpicxx
```

With the CUDA Toolkit available, the measured A40 target can be built with:

```bash
make gpu CUDA_ARCH=sm_86
```

For another GPU, select its supported architecture. Use the same MPI implementation for compilation and execution. The Vanda scripts load the required modules and use isolated build directories. See [docs/VANDA_RUN.md](docs/VANDA_RUN.md) for submission instructions and [docs/VANDA_EIGEN.md](docs/VANDA_EIGEN.md) for eigenvalue details.

The optional GPU Lanczos build has its own target and output subdirectory:

```bash
make gpu-lanczos CXX=g++ MPICXX=mpicxx CUDA_ARCH=sm_86
```

The module accepts `CUDA_HOME` and `CUDA_LIBDIR`; the Vanda PBS script selects the actual toolkit paths. Its default binaries are isolated from the PageRank build.

The project implements its numerical kernels and projected solver directly; it does not use BLAS/LAPACK solver libraries. Existing results can be analyzed without rebuilding or rerunning completed experiments.
