# Vanda MPI+OpenMP Eigenvalue Experiment with MPICH

The primary entry point is `scripts/run_vanda_eigen_mpich.pbs`; `scripts/run_vanda_eigen.pbs` contains the same workflow. Both run the C++ MPI+OpenMP solvers.

## C++ computation and Python orchestration

`src/power_symmetric.cpp` and `src/lanczos.cpp` implement the numerical methods. `src/symmetric.hpp` contains MPI collectives and OpenMP loops. Python launches processes, applies timeouts, checks output, performs independent small-graph checks, and organizes CSV files. A Python subprocess timeout means the launched program did not exit within its limit; it does not mean the numerical method was implemented in Python.

Both solvers use the same lazy symmetric graph operator `B=(I+S)/2`, with `S` formed from the symmetrized, normalized input graph. The shared seed defines the same initial vector. This eigenproblem is separate from directed PageRank with damping 0.85. Lanczos uses full reorthogonalization and a handwritten projected eigensolver.

## MPICH launch configuration

The workflow loads `mpich/ge/gcc/64/4.1.1` and selects its compiler and launcher explicitly through `MPI_HOME`, whose recorded installation is `/cm/shared/apps/mpich/ge/gcc/64/4.1.1`. Do not reuse binaries built with OpenMPI.

The PBS request is one node, 8 CPU cores, 4 MPI slots, 16 GB, and two hours, without a GPU, in `auto_free`. The script checks the allocation before starting MPICH Hydra's local fork launcher on the allocated node. The launcher options are:

```text
-rmk user -launcher fork -hosts localhost -bind-to core:2
```

The measured layout is four ranks with two OpenMP threads per rank. `HYDRA_TOPO_DEBUG` is unset because that debugging option prints binding information without performing the normal binding action.

The default OFI request is `FI_PROVIDER=sockets`, avoiding an RDMA-device requirement in this single-node workflow. MPI processes and collectives are still executed. Override the request explicitly when submitting if needed:

```bash
qsub -v FI_PROVIDER=sockets scripts/run_vanda_eigen_mpich.pbs
```

Current submission scripts write the requested communication environment to `communication_environment.json`; the recorded sampling driver remains the version used for job 1526107, whose original provenance does not include later-added fields. Configuration requests do not prove the transport used by every message or tuned cross-node RDMA performance.

References: [libfabric sockets provider](https://github.com/ofiwg/libfabric#sockets) and [MPICH Hydra process management and binding](https://github.com/pmodels/mpich/blob/main/doc/wiki/how_to/Using_the_Hydra_Process_Manager.md). If the installed launcher or required options are unavailable, the workflow stops and preserves diagnostics.

## Workflow for a new cluster run

1. Build fresh serial and MPI binaries in `work/eigen_mpich_build_JOBID/bin` using the selected MPICH compiler.
2. Probe one-rank and four-rank MPI execution with two OpenMP threads per rank. The probe exercises initialization, OpenMP, Allreduce, Allgatherv, empty partitions, Barrier, and Finalize, and checks the actual MPI library version.
3. Run 48 small-graph serial/MPI evaluations using `scripts/validate_eigen_pair.py`. Both algorithms use the same operator and initial seed; independent dense calculations check their residuals.
4. After validation, sample `power_symmetric_mpi` and `lanczos_mpi` with four ranks and two threads per rank. The driver rejects a mismatched reported layout.

The formal experiment uses uniform, hub, and community graphs, each with `N=100000`, input `E=800000`, and seed `20261029`. Power budgets are 10/20/50/100/200/500; Lanczos budgets are 5/10/20/30/50. Five repeats per configuration give 33 configurations and 165 measurements. Timing repeats are not independent graph samples.

This experiment compares accuracy, work, and numerical solve time at a fixed MPI+OpenMP layout. It does not provide serial performance baselines or CPU-core scaling curves. Group-wide PageRank strong and weak scaling uses Member A's experiments.

## Validation evidence and local audit

The package now includes both `validation/summary.json` and the recovered `validation/comparisons.csv` under `results/Vanda_eigen_1526107/`. All 48 CSV records match the original validation site, node, job ID, and measurement ID. This stage is distinct from the 165 performance measurements.

The original validator constructs small dense matrices independently of the sparse solver. It evaluates `||Bq-lambda*q||_2` from the dumped vector and compares it with the solver's reported residual. It also checks vector normalization. The archived driver uses `1e-10` for these consistency checks. Their numerical differences cannot be reconstructed from the CSV alone because the vector dumps and solver-reported residuals were not retained; the original suite PASS remains the evidence for them.

The recovered CSV does retain the Ritz eigenvalues and independent dense residuals, allowing these endpoint checks to be reconstructed:

- For the zero-edge and single-vertex graphs, `B=I/2`, so the expected eigenvalue is 0.5. For the other test graphs, the expected largest eigenvalue is 1.
- Endpoint acceptance requires both absolute eigenvalue error and dense residual below `1e-8`, using the archived validator's criteria. Endpoint rows have budget greater than one, or `N=1`. A general one-step evaluation is not required to converge.
- Serial power/Lanczos one-step pairs compare their eigenvalue and dense-residual outputs using the archived `1e-12` criterion.
- Four MPI ranks on the zero-edge, single-vertex, and tiny-edge cases exercise empty row partitions. The lazy shift avoids the equal-magnitude positive/negative eigenvalue ambiguity on bipartite graphs.

The table reports maxima across both algorithms and serial/MPI endpoint rows:

| Test graph | N | Expected eigenvalue | Maximum eigenvalue error | Maximum dense residual | Threshold for each | Recomputed result |
|---|---:|---:|---:|---:|---:|---|
| Zero-edge | 3 | 0.5 | 5.5511e-17 | 6.2450e-17 | < 1e-8 | PASS |
| Single vertex | 1 | 0.5 | 0 | 0 | < 1e-8 | PASS |
| Tiny edge | 2 | 1 | 2.2204e-16 | 1.5701e-16 | < 1e-8 | PASS |
| Path | 12 | 1 | 2.2204e-16 | 1.0180e-10 | < 1e-8 | PASS |
| Cycle | 10 | 1 | 2.2204e-16 | 9.1633e-11 | < 1e-8 | PASS |
| Self-loop and isolated vertex | 5 | 1 | 4.4409e-16 | 9.5743e-11 | < 1e-8 | PASS |

Across six serial one-step power/Lanczos pairs, the largest recorded eigenvalue difference is 0 and the largest dense-residual difference is approximately `1.1102e-16`, below `1e-12`. Recorded zeros describe stored floating-point results. The power solver requests a `1e-10` stopping tolerance at the larger budget; this is distinct from `1e-8` validation acceptance. The path case exhausts its 1000-step budget with residual about `1.0180e-10` and still passes the latter check.

From the project root, audit the existing evidence locally:

```bash
python3 scripts/audit_eigen_validation.py
```

This standard-library script checks CSV identity, all six cases and their expected configurations, recorded source hashes, endpoint accuracy, and serial one-step consistency. It prints JSON to standard output and does not launch solvers or write measurements. Its thresholds come from the archived validation driver. The original summary did not record that driver's hash; the local audit records the hash of the archived copy and distinguishes recomputed checks from the original suite PASS.

To perform a new validation on Vanda instead, use the PBS workflow above. A new run must use its own job ID and fresh result directory. It must not be relabeled as job 1526107.

## Reproducing the full experiment

Extract the complete package into a fresh Vanda directory and submit from the project root:

```bash
cd ~/graph-ranking-clean/graph-ranking-hpc-c
qsub -v FI_PROVIDER=sockets scripts/run_vanda_eigen_mpich.pbs
```

Existing results do not need a rerun. For a new job, replace `JOBID` with its numeric ID:

```bash
tail -n 50 results/Vanda_eigen_JOBID/workflow.log
```

The log should start with `START mode=MPI+OpenMP implementation=MPICH` and use `mpiexec.hydra`. After the probe, expect `MPI_EIGEN_STARTUP_PASS`, followed by `VALIDATION_START`/`VALIDATION_PASS` and measurement progress. Normal completion records `EIGEN_COMPLETE` and `SUCCESS`. Do not use a previous job's log to assess a new run.

## Output files and performance plots

A full new workflow writes `results/Vanda_eigen_JOBID/` with:

- `mpi_launcher_info.txt` and `mpi_compiler_command.txt`: launcher and compiler details.
- `mpi_probe/`: diagnostic logs and actual MPI library information.
- `validation/summary.json`, `validation/comparisons.csv`, and `validation/raw/`: correctness records.
- `comparison/runs.csv`, `comparison/summary.csv`, and `comparison/provenance.json`: timings, residuals, identities, hashes, and execution configuration.
- `pbs_job.txt`, `pbs_nodefile.txt`, and `cpu_topology.txt`: allocation and node records.

Retain these original files for review. Binding configuration alone is not independent proof of exclusive CPU placement.

For the included job 1526107 measurements, generate figures locally from the project root:

```bash
python3 -m pip install -r requirements-visualization.txt
python3 scripts/visualize_vanda.py eigen \
  --runs results/Vanda_eigen_1526107/comparison/runs.csv \
  --out figures/Vanda_eigen_1526107
```

For a future job, download its `comparison/` CSV/JSON files and `validation/` evidence into a separate directory before local analysis. Use performance `runs.csv` for plots and validation `comparisons.csv` for correctness checks.

Figures compare direct residual, actual SpMV count, and complete `solve_s` at the fixed layout. Lanczos's time includes full reorthogonalization, the projected solve, reconstruction, and the final direct residual. Its final residual requires one additional SpMV. Graph loading and MPI startup are excluded. Displayed zero residuals are floored at `1e-17` on logarithmic axes without modifying the raw data.

## Retained evidence and limitations

The package contains five original files from job 1526107: three performance records and two validation records. All retain their supplied bytes. Source/driver hashes and CSV identities are checked during packaging and local auditing. No MPI execution, new numerical experiment, or plotting was performed during this documentation update.

Remote probe logs, PBS allocation records, original graph files, vector dumps, and binaries were not supplied. The previously missing per-evaluation CSV is now included; these other absent artifacts have not been reconstructed or invented.
