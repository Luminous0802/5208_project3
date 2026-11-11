# Reproducing the Experiments on Vanda

The commands below are for an optional new reproduction run. Existing measurements can be analyzed locally without resampling. Work in a fresh project/output directory rather than overwriting an active job.

## Prepare a fresh directory

Upload `graph-ranking-hpc-c-clean.zip`, then extract it on the Vanda login node:

```bash
mkdir -p ~/graph-ranking-clean
unzip ~/graph-ranking-hpc-c-clean.zip -d ~/graph-ranking-clean
cd ~/graph-ranking-clean/graph-ranking-hpc-c
```

## GPU and 12-core CPU comparison

```bash
qsub -l walltime=04:00:00 scripts/run_vanda_gpu_full.pbs
```

The script requests one GPU and 12 CPU cores in `auto_free`. Its modules are `gcc11/11.3.0`, `cuda12.4/toolkit/12.4.1`, `openmpi/gcc/64/4.1.8`, and `python39`. Check module availability and resource policies in the target environment before a new reproduction run.

The workflow builds and validates the implementations, checks graph SHA256 hashes against Member B's specification, and samples six graphs x (one serial + four hybrid + two CUDA configurations) x five repeats, giving 210 measurements. Outputs go to `results/Vanda_gpu_JOBID/`; plotting is performed later on the local computer. Existing job 1520880 results remain available and need no rerun for this package update.

## MPI+OpenMP power iteration and Lanczos

```bash
qsub -v FI_PROVIDER=sockets scripts/run_vanda_eigen_mpich.pbs
```

This uses MPICH 4.1.1 and requests one node, 8 CPU cores, 4 MPI slots, 16 GB, and two hours, without a GPU. The measured layout is 4 MPI ranks x 2 OpenMP threads. It is independent of the group's CPU scaling workflows.

Job 1526107 performance results and all 48 per-evaluation validation records are included under `results/Vanda_eigen_1526107/`. A new reproduction uses a new job ID and output directory. See [VANDA_EIGEN.md](VANDA_EIGEN.md) for numerical and timing definitions.

## GPU Lanczos with same-node CPU baselines

```bash
qsub scripts/run_vanda_gpu_lanczos.pbs
```

This separate workflow requests one GPU, four CPU cores, two MPI slots, 16 GB, and two hours. It loads CUDA 12.4 and MPICH 4.1.1, validates 48 small-graph evaluations, then runs 300 timing measurements. Two MPI GPU processes share one device. Optional pilot mode is `qsub -v GPU_LANCZOS_STAGE=pilot scripts/run_vanda_gpu_lanczos.pbs`.

Completed job 1536096 evidence is included under `results/Vanda_gpu_lanczos_1536096/`; no rerun is needed for local review. New workflows write `results/Vanda_gpu_lanczos_JOBID/experiment/`. Details are in [the GPU Lanczos README](../gpu_lanczos/README.md).

## Status and logs

```bash
qstat -f JOBID
# After the job has finished:
qstat -xf JOBID
```

GPU workflow logs use `results/Vanda_gpu_JOBID/workflow.log`; eigenvalue logs use `results/Vanda_eigen_JOBID/workflow.log`. Replace `JOBID` with the actual numeric job ID and use the actual project directory. PBS jobs can continue after SSH disconnects; completion must be checked from exit status and final workflow markers.

## Local analysis of existing evidence

No cluster allocation is needed for the validation audit:

```bash
python3 scripts/audit_eigen_validation.py
python3 scripts/audit_gpu_lanczos.py
```

Use the main README's `visualize_vanda.py gpu` and `eigen` commands for performance plots. These read the existing performance CSV files. Duplicate Part C CPU scaling submission entry points have been removed; the final group report uses Member A's scaling results.
