# GPU Lanczos on Vanda

This module is integrated into the Part C package. The recorded full experiment completed as Vanda job **1536096.stdct-mgmt-02** on **GN-A40-003**, using one NVIDIA A40. The retained evidence contains 300 performance measurements, 60 configurations with five repeats each, and 48 small-graph validation records. The imported implementation's complete 15-file source fingerprint matches the recorded `source_sha256`.

The source archive was originally prepared before the run, so its historical static-check record says the cluster experiment had not yet been executed. That record is preserved in `../docs/GPU_LANCZOS_SOURCE_IMPORT.json`; the completed CSV evidence is retained separately. This integration performs local file/CSV checks, not a new cluster compilation or measurement.

## Mathematical problem and implementations

All variants solve the same FP64 symmetric lazy eigenproblem `B=(I+S)/2`, with two-pass full reorthogonalization, Ritz-vector reconstruction, and a direct residual. The directed input is symmetrized and deduplicated before constructing the normalized symmetric operator `S`. This differs from directed PageRank with damping 0.85.

The three graph families are uniform, hub, and community, each with `N=100000`, input `E=800000`, and seed `20261029`. `configs/graphs.json` records the generator and input SHA256 hashes matching the earlier CPU eigenvalue job 1526107. Timing repeats are not independent graph realizations.

| Variant | MPI processes | Host threads per process | GPU use |
|---|---:|---:|---|
| `cpu_serial` | 1 | 1 | None; serial reference |
| `cpu_mpi2_omp2` | 2 | 2 | None; four-core CPU baseline |
| `gpu_mpi1` | 1 | 2 | One GPU |
| `gpu_mpi2_shared` | 2 | 2 | Both processes share the same GPU |

The GPU executes sparse matrix-vector multiplication, dot products/norms, vector updates, two-pass full reorthogonalization, and Ritz-vector reconstruction. The small projected eigenproblem is solved on the host and is included in `solve_s`. MPI exchanges host buffers with Allgatherv/Allreduce; GPU pointers are not passed to MPI, so CUDA-aware MPI is not required.

GPU processes respect `CUDA_VISIBLE_DEVICES`. The driver checks the host and device UUID and requires one visible GPU. The shared variant is not a multi-GPU scaling experiment. Additional MPI processes are not assumed to improve performance.

## Retained results and local checks

The completed downloaded records are stored under `../results/Vanda_gpu_lanczos_1536096/`:

- `runs.csv`: 300 timing measurements.
- `summary.csv`: original 60-row summary.
- `validation/comparisons.csv`: all 48 validation records.

From the project root, audit the existing records and verify the imported source fingerprint:

```bash
python3 scripts/audit_gpu_lanczos.py
```

This standard-library command does not launch MPI/CUDA, allocate a GPU, or generate plots. It checks identities, inputs, configuration coverage, summaries, selected budgets, validation diagnostics, and source hashes. Original vector dumps, binary files, full provenance JSON, allocation logs, and binding logs were not supplied; their independent verification is not implied by the retained CSV status fields.

At the smallest common passing tested budgets, median numerical solve times are:

| Graph | Budget | CPU 2x2 (ms) | GPU 1 process (ms) | Shared GPU 2 processes (ms) | CPU/GPU-1 ratio |
|---|---:|---:|---:|---:|---:|
| Uniform | 20 | 59.253 | 13.699 | 39.672 | 4.33 |
| Hub | 20 | 61.743 | 20.043 | 45.729 | 3.08 |
| Community | 30 | 104.806 | 22.509 | 61.018 | 4.66 |

All five repeats of all four variants reach residual below `1e-8` at these budgets. Broader CPU/GPU-1 application-time ratios are only 0.87--1.07. The measured solve-time improvement therefore does not imply consistent end-to-end acceleration. Two processes sharing the A40 are slower than one process at these selected budgets. The baseline is from the same GPU-node batch, not the earlier IN-001 CPU experiment.

## Reproduce on Vanda

From the root of the combined `graph-ranking-hpc-c` package:

```bash
qsub scripts/run_vanda_gpu_lanczos.pbs
```

The PBS script requests one node, four CPU cores, two MPI slots, 16 GB host memory, one GPU, and two hours in `auto_free`. It loads GCC 11.3, CUDA 12.4, MPICH 4.1.1, and Python 3.9, then uses MPICH Hydra's single-node fork launcher and a default `FI_PROVIDER=sockets` request. The CUDA object is compiled with `nvcc`, and the MPI host driver is compiled and linked with `mpicxx`.

The script detects the CUDA architecture and toolkit library directory, verifies the allocation, and builds into `work/gpu_lanczos_build_JOBID/bin`. It performs compilation, 48 validation evaluations, graph-hash checks, and the full benchmark. A failure stops the workflow and preserves diagnostics.

The full plan is three graphs x five budgets (5/10/20/30/50) x four variants x five repeats: 300 measurements. For an optional startup/validation pilot:

```bash
qsub -v GPU_LANCZOS_STAGE=pilot scripts/run_vanda_gpu_lanczos.pbs
```

The pilot uses the same 48 validation evaluations but only eight timing runs: uniform input, budgets 5/10, four variants, one repeat. Pilot timings are not the full five-repeat benchmark.

Fresh jobs write `results/Vanda_gpu_lanczos_JOBID/experiment/` with their own job ID. The bundled downloaded records are stored directly under `results/Vanda_gpu_lanczos_1536096/`; this is an archival layout, not a change to the job script's output path. Existing results need no rerun for this integration.

An optional manual build is available from the project root:

```bash
make gpu-lanczos CXX=g++ MPICXX=mpicxx CUDA_ARCH=sm_86
```

This uses a separate `bin/gpu_lanczos/` directory. The module Makefile also accepts `CUDA_HOME` and `CUDA_LIBDIR`. Use matching MPICH compiler/runtime paths and the correct toolkit location; the PBS entry handles these settings automatically.

## Validation design

Six small graphs are tested at one step and a sufficient budget with all four variants, giving 48 solver invocations. The two budgets coincide at one for the single-vertex case. Cases cover zero edges, one vertex, a tiny edge, an uneven path, a cycle, self-loop/disconnected vertices, uneven MPI partitions, and empty partitions.

The independent Python dense calculation checks vector norms, Rayleigh quotients, direct residuals, known largest eigenvalues, and basis orthogonality. One-step vectors are compared with the serial vector after sign alignment. Converged eigenvectors are not required to agree elementwise when repeated eigenvalues make them nonunique.

The driver defines these thresholds in `scripts/lanczos_common.py`:

| Check | Threshold |
|---|---:|
| Vector normalization error | 1e-10 |
| Dense versus reported residual difference | 1e-10 |
| Rayleigh quotient versus reported eigenvalue | 1e-10 |
| Eigenvalue error / full-budget residual | 1e-8 |
| Basis orthogonality error | 1e-9 |
| One-step vector L1 difference | 1e-9 |
| One-step vector Linf difference | 1e-10 |

The retained CSV reports PASS for all 48 evaluations. Across the 24 full-budget records, the largest dense residual is approximately `7.8945e-16`; the largest one-step L1 difference is approximately `4.1980e-16`. These are recorded results, not newly executed validation. The three supplied CSV files do not include the original `thresholds.json`, vectors, or per-process logs; the matching source defines the protocol.

## Timing definitions

- `iteration_s`: iteration wall time, including host/device transfers, MPI, and waiting.
- `solve_s`: iteration, host projected eigensolve, Ritz reconstruction, and direct residual; excludes graph loading, GPU allocation, and warmup.
- `end_to_end_s`: application interval after MPI initialization through solve completion, including loading, GPU setup, and warmup; excludes external launch/exit, event readback, and vector output.
- H2D/D2H fields: blocking host-observed transfer intervals, which may include queueing. Separately maximized process times are not additive communication percentages.
- GPU-event fields: accumulated per-kernel CUDA-event intervals, then maximized across processes. Shared-device scheduling can affect these intervals; they are not exclusive-device pure kernel time. Event readback occurs after the solve timer stops.
- Warmup covers the computational kernels, then restores the initial state. It is charged to the application interval, not iteration/solve time.

The accuracy target is evaluated after the fixed budget, rather than used for early stopping. Natural invariant-subspace breakdown may end the iteration earlier; actual step and SpMV counts are retained. Speedup calculations require matching job, input, precision, budget, actual steps, and SpMV count. Medians summarize five randomized/interleaved repetitions; min/max describe observed ranges rather than confidence intervals.

## Logs and downloads for a future run

Replace `JOBID` with the numeric PBS job ID:

```bash
qstat -f JOBID
tail -n 40 results/Vanda_gpu_lanczos_JOBID/workflow.log
qstat -xf JOBID
```

Successful validation records `GPU_LANCZOS_VALIDATION_PASS`; full timing progress then reaches 300 runs. Completion requires `GPU_LANCZOS_COMPLETE` and the final `SUCCESS` marker. SSH disconnects do not determine PBS job status.

For future runs, retain `experiment/runs.csv`, `experiment/summary.csv`, `experiment/provenance.json`, and the full `experiment/validation/` directory, together with PBS, compiler, device, binding, and raw process logs. Download them to a separate local folder before analysis.

## Local visualization

From the combined project's root:

```bash
python3 -m pip install -r requirements-visualization.txt
python3 gpu_lanczos/scripts/analyze_gpu_lanczos.py \
  results/Vanda_gpu_lanczos_1536096/runs.csv \
  --out figures/Vanda_gpu_lanczos_1536096/summary.csv \
  --plot-dir figures/Vanda_gpu_lanczos_1536096/plots
```

This reads existing measurements, validates their schema, and generates timing/accuracy figures using logarithmic axes. Displayed residuals are floored at `1e-17` without changing input data. Use fresh output paths. No plots were generated during this integration.
