# Vanda Job 1520880: GPU and 12-Core CPU Comparison

This directory contains two original files downloaded from Vanda:

- `runs.csv`: 210 measurements.
- `summary.csv`: 42 configuration summaries, five repeats each.

Both files retain their original bytes. The recorded site is Vanda, job `1520880.stdct-mgmt-02`, node GN-A40-004, with one NVIDIA A40, FP64, and 50 iterations. The six graph hashes match Member B's input manifest. Measurements comprise 30 serial CPU, 120 MPI+OpenMP CPU, and 60 CUDA runs. Four hybrid configurations use either 2 ranks x 6 threads or 6 ranks x 2 threads with the tested partition/scheduling choices.

`IMPORT_RECORD.json` records local file/hash, summary, matched-baseline, graph, and source checks. It does not represent a rerun of remote validation. Original GPU preflight JSON, device metadata, PBS allocation records, and binding logs were not supplied.

## Correctness evidence

The performance CSV records full-vector L1 comparisons against the same-node FP64 serial reference. Maximum differences are approximately `1.1532e-15` across 60 CUDA runs and `9.1576e-18` across 120 hybrid runs. The maximum absolute rank-sum deviation across all 210 runs is approximately `9.0379e-12`. All recorded runs complete 50 iterations. Full-vector agreement is stronger evidence than rank-sum conservation alone.

## Timing scope

`iteration_s` is host-observed loop time. `gpu_timeline_s` is a CUDA-event device interval. H2D/D2H, GPU setup/allocation, warmup, and internal `end_to_end_s` are recorded separately. `transfer_inclusive_s` is H2D + loop + D2H and excludes setup and warmup; it is not full end-to-end time. `launcher_wall_s` includes additional launch/wait/validation overhead and is not a uniform algorithm timing scope.

These records are stored separately from the power/Lanczos results to avoid filename collisions. The two experiments use different operators and timing definitions and are analyzed separately.

## Local plots

From the project root:

```bash
python3 -m pip install -r requirements-visualization.txt
python3 scripts/visualize_vanda.py gpu \
  --runs results/Vanda_gpu_1520880/runs.csv \
  --out figures/Vanda_gpu_1520880
```

This reads existing CSV data without running solvers. No figures were generated during this documentation update. A different axis scale can be selected in the local plotter without new measurements.
