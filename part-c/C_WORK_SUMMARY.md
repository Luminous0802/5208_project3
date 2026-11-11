# Part C Contributions and Vanda Experiment Record

Documentation updated: 10 November 2026. This update incorporates the recovered per-evaluation validation CSV; it does not represent a new cluster experiment.

This document summarizes my Part C implementations, completed Vanda measurements, and result preparation. Numerical measurements were made on NUS Vanda. Figures can be generated locally from the original CSV files. No solver, benchmark, or plotting run was performed during this documentation update.

## 1. Contribution scope

My work covers GPU PageRank, comparisons against serial and 12-core MPI+OpenMP CPU baselines on the GPU node, and a power-iteration/Lanczos comparison under MPI+OpenMP. I also organized batch submission, correctness checks, timing definitions, original evidence, and local analysis tools for the group report.

PageRank conventions, graph specifications, and CPU references build on the corresponding A/B contributions. Hybrid partition/scheduling configurations establish matched CPU baselines on the GPU node; their earlier implementation is not counted again as an independent Part C contribution.

Member A handles group-wide strong and weak scaling. Duplicate Part C CPU scaling workflows have been removed from this package, while the GPU experiment's 12-core CPU comparisons remain.

## 2. Completed experiments

| Experiment | Vanda job | Main resources | Configurations | Repeats | Timing measurements |
|---|---|---|---:|---:|---:|
| GPU PageRank / CPU comparison | 1520880.stdct-mgmt-02 | GN-A40-004, one NVIDIA A40, 12 CPU cores | 42 | 5 | 210 |
| Power iteration / Lanczos | 1526107.stdct-mgmt-02 | IN-001, 4 MPI ranks x 2 OpenMP threads | 33 | 5 | 165 |
| CPU/GPU Lanczos | 1536096.stdct-mgmt-02 | GN-A40-003, one A40, same-node serial/2x2 CPU references | 60 | 5 | 300 |

Repeats measure the same configuration on the same graph; they are not independent random graph samples. The timing counts exclude the separate small-graph validation stage.

## 3. GPU PageRank

### Algorithm and inputs

The GPU and matched CPU programs use damping 0.85, a uniform initial vector, uniform redistribution of dangling mass, the shared directed edge-list format, and FP64. Performance runs execute exactly 50 iterations. CPU/GPU inputs are matched by SHA256 against Member B's specifications.

The six graphs use seed `20261029`. Four B3 graphs have `N=500000`, `E=4000000`: uniform, hub, community ordered, and community shuffled. Two B4 graphs have `N=1000000`, `E=8000000`: uniform and hub. Structure comparisons use matched sizes; absolute differences between B3 and B4 are not attributed solely to structure.

### CUDA mappings

The pull update has two implementations:

- **Thread-per-vertex:** one CUDA thread traverses one destination's incoming edges.
- **Warp-per-vertex:** one warp cooperatively traverses a destination row and reduces its contributions.

GPU kernels also reduce dangling mass and residuals. Measurements use block size 256 and two warmup iterations, reset the initial vector, then time 50 iterations. The recorded runtime is CUDA 12.4, with an O3/C++17/FP64/`sm_86` build tag. The experiment tests these mappings on the selected graphs; it does not include multi-GPU scaling.

### Same-node CPU references

In addition to the serial reference, four 12-core hybrid configurations were tested:

| MPI ranks | OpenMP threads per rank | Partition | Schedule |
|---:|---:|---|---|
| 2 | 6 | vertex | static |
| 6 | 2 | vertex | static |
| 6 | 2 | edge | static |
| 6 | 2 | edge | dynamic, chunk 64 |

The launcher requests `--map-by slot:PE={threads} --bind-to core`, and the driver checks CPU masks and the reported rank/thread layout. Original binding logs were not supplied, so the package does not claim an independently archived affinity audit. CPU and CUDA programs run in turn; one GPU is compared with 12 CPU cores, not 12 GPUs. The serial baseline is the same-node `pagerank_reference`.

### Correctness and timing

The workflow defines a small-graph preflight stage, but its original JSON/logs are not included. The retained performance CSV provides complete-vector comparisons with the FP64 serial reference:

| Check | Maximum recorded error |
|---|---:|
| CUDA versus serial vector L1 difference, 60 rows | 1.1532e-15 |
| Hybrid versus serial vector L1 difference, 120 rows | 9.1576e-18 |
| Absolute rank-sum deviation across 210 rows | 9.0379e-12 |

All recorded runs complete 50 iterations. Full-vector comparisons provide stronger correctness evidence than rank-sum conservation alone. Sanitizer success is not claimed without the missing logs.

| Field | Timing meaning |
|---|---|
| `iteration_s` | Host-observed loop time; CPU loops include their communication |
| `gpu_timeline_s` | CUDA-event interval, not a sum of individually profiled kernels |
| `h2d_s`, `d2h_s` | Host/device transfers |
| `gpu_setup_s`, `warmup_s` | GPU initialization/allocation and warmup |
| `transfer_inclusive_s` | H2D + loop + D2H, excluding setup and warmup |
| `end_to_end_s` | Implementation-defined internal interval; see source boundaries |
| `launcher_wall_s` | External launch/wait interval, potentially including validation |

Comparisons use five-run medians and retain min/max and the original samples. Loop, transfer-inclusive, and internal end-to-end comparisons are kept distinct.

### Observed GPU results

Thread-per-vertex has the lower loop median on all six graphs. Times below are host-observed loop milliseconds for 50 iterations:

| Graph | GPU thread | GPU warp | Thread loop speedup over best tested 12-core hybrid |
|---|---:|---:|---:|
| B3 uniform | 27.372 | 108.062 | 4.55x |
| B3 hub | 65.199 | 109.220 | 4.17x |
| B3 community ordered | 20.734 | 108.976 | 4.26x |
| B3 community shuffled | 27.475 | 108.058 | 4.55x |
| B4 uniform | 82.784 | 190.258 | 3.57x |
| B4 hub | 77.639 | 200.952 | 7.91x |

The reference is the lowest loop median among the four tested hybrid configurations. These are loop ratios, not end-to-end speedups or bounds against every possible CPU optimization. Degree statistics and profiling would be needed to establish the causes of the mapping differences.

Sources: [GPU measurements](results/Vanda_gpu_1520880/runs.csv) and [original summary](results/Vanda_gpu_1520880/summary.csv).

## 4. MPI+OpenMP power iteration and Lanczos

### Purpose and implementation

The methods are compared on the same symmetrized normalized operator `S`, shared lazy operator `B=(I+S)/2`, initial seed, and parallel layout. This differs mathematically from the damped directed PageRank task and is analyzed separately.

Both methods have serial and MPI+OpenMP C++ implementations. Formal performance sampling uses `power_symmetric_mpi` and `lanczos_mpi`. MPI provides collectives; OpenMP handles local computation. Python orchestrates processes, independent checks, logs, and CSV records.

The successful recorded layout uses MPICH 4.1.1, four ranks, and two threads per rank on one node, with Hydra's fork launcher and an `FI_PROVIDER=sockets` environment request. This remains real MPI execution. The request is not proof of every message's transport or tuned cross-node RDMA performance.

### What validation did

The original summary records overall PASS for 48 solver evaluations. The recovered `validation/comparisons.csv` contains those 48 records with the same validation identity. Six small cases cover zero edges, one vertex, a tiny edge, a path, a cycle, and self-loop/isolated vertices. Both methods are tested serially and with four MPI ranks, at one-step and larger budgets.

The validator independently constructs a dense operator, evaluates Ritz-pair residuals, checks expected eigenvalues, compares one-step results from the shared initial vector, and exercises empty MPI partitions. The lazy shift avoids equal-magnitude positive/negative eigenvalue ambiguity on bipartite cases.

The recovered endpoint records satisfy both absolute eigenvalue error and dense residual below `1e-8` using the archived validator's criteria. The largest endpoint residual is approximately `1.0180e-10`. Across six serial one-step pairs, the largest recorded eigenvalue difference is zero and residual difference is approximately `1.1102e-16`, below the archived `1e-12` criterion. Endpoint selection excludes generic one-step rows from a convergence requirement; `N=1` is included directly.

Run `python3 scripts/audit_eigen_validation.py` to reproduce these checks from existing evidence. The CSV has no vector dumps or solver-reported residuals, so vector-normalization errors and dense-versus-reported residual differences cannot be reconstructed; those checks retain the original suite PASS evidence. The original summary did not hash the validation driver itself. [docs/VANDA_EIGEN.md](docs/VANDA_EIGEN.md#validation-evidence-and-local-audit) provides the six-case numerical table and threshold provenance.

### Formal performance sampling

- Graph families: uniform, hub, community; each `N=100000`, input `E=800000`, seed `20261029`.
- Power budgets: 10, 20, 50, 100, 200, 500.
- Lanczos budgets: 5, 10, 20, 30, 50, with full reorthogonalization.
- 33 configurations, five repeats each, giving 165 measurements.
- Accuracy target: direct residual below `1e-8`; nonpassing small budgets remain valid observations.

`solve_s` includes numerical iteration, Lanczos projected solving/reconstruction, and direct residual evaluation. It excludes graph loading and MPI startup. Lanczos's final residual requires an extra SpMV, included in the work count.

### Observed eigenvalue results

The table selects the smallest tested budget at which all five repeats pass the residual target. Times are median `solve_s` in milliseconds:

| Graph | Power budget | Power time | Lanczos budget | Lanczos time |
|---|---:|---:|---:|---:|
| uniform | 100 | 89.334 | 20 | 35.242 |
| hub | 100 | 157.382 | 20 | 49.404 |
| community | 500 | 429.506 | 30 | 59.229 |

Lanczos reaches the target at a smaller tested budget and lower solve time in these selections. Discrete budget sampling does not establish the exact minimum iteration count or minimum time to tolerance. There is no corresponding serial performance sweep, so these ratios are not MPI parallel speedups.

Sources: [performance records](results/Vanda_eigen_1526107/comparison/runs.csv), [summary](results/Vanda_eigen_1526107/comparison/summary.csv), [provenance](results/Vanda_eigen_1526107/comparison/provenance.json), [validation summary](results/Vanda_eigen_1526107/validation/summary.json), and [validation records](results/Vanda_eigen_1526107/validation/comparisons.csv).

### Supplementary GPU Lanczos

The GPU implementation is included under `gpu_lanczos/`, with a separate MPICH/CUDA PBS workflow. CUDA computes sparse products, dot products, vector operations, two-pass full reorthogonalization, and Ritz reconstruction. The projected solve remains on the host; MPI uses host buffers. Its 15-file source fingerprint matches the new timing CSV. Shared CPU source files and helpers are unchanged.

Job 1536096 completed 300 measurements on GN-A40-003, comparing serial CPU, 2x2 CPU, one-process GPU, and two-process shared-GPU execution on three graph families and five budgets. At common passing budgets 20/20/30 for Uniform/Hub/Community, CPU-2x2/GPU-1 solve ratios are 4.33/3.08/4.66. Broader application ratios span 0.87--1.07; two processes sharing one A40 are slower than one at these selections. These matched comparisons are separate from the earlier CPU Power/Lanczos ratios.

The retained validation CSV records 48 passing evaluations with small graphs, one-step vector comparisons, dense residual checks, Rayleigh/normalization/orthogonality diagnostics, uneven partitions, and empty partitions. Maximum full-budget dense residual is approximately 7.8945e-16 and one-step L1 difference approximately 4.1980e-16. The original full vectors, allocation/binding logs, binaries, and runtime provenance JSON were not supplied.

Use `python3 scripts/audit_gpu_lanczos.py` for existing evidence; submit `scripts/run_vanda_gpu_lanczos.pbs` only for a fresh reproduction. [Module instructions](gpu_lanczos/README.md) and [retained results](results/Vanda_gpu_lanczos_1536096/README.md) describe the protocol and timing scopes.

## 5. Deliverables and integration

The package contains source, GPU/MPICH submission workflows, the three completed experiments, the recovered validation records, and local analysis tools. GPU and eigenvalue results use separate directories to avoid filename collisions. Original CSV/JSON evidence is unchanged; import records and checksums describe file checks rather than new experiments.

A/B/GPU/eigenvalue summary integration remains explicit. No duplicate Part C CPU scaling workflows, historical local timing data, binaries, or pre-generated figures are included. Group-wide reporting should combine the A/B results with the matched C comparisons after checking graph identity, precision, iteration rules, hardware, and timing scope.

The supplied data can be analyzed without a new GPU allocation. Preserve any remaining remote PBS, binding, probe, and vector logs if available. The package distinguishes the recovered eigenvalue CSV from other artifacts that remain absent. See [README.md](README.md) for local commands and [docs/PACKAGE_SCOPE.md](docs/PACKAGE_SCOPE.md) for the current package scope.
