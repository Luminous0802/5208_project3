# Recorded Vanda Results

This package retains three completed Vanda experiments:

- `Vanda_gpu_1520880/`: 210 GPU/12-core CPU measurements, 42 configurations, and the original performance summary.
- `Vanda_eigen_1526107/`: 165 power/Lanczos measurements, 33 configurations, performance provenance, the original 48-evaluation validation summary, and the recovered `validation/comparisons.csv`.

- `Vanda_gpu_lanczos_1536096/`: 300 CPU/GPU Lanczos measurements, 60 configurations, and 48 separate small-graph validation records.

All ten original result files retain their supplied bytes. Historical `IMPORT_RECORD.json` files describe packaging checks rather than new experiments. The recovered validation CSV has its own import record under `Vanda_eigen_1526107/validation/`.

The 48 validation records and 165 eigenvalue timing records belong to different stages with different measurement identifiers. `comparisons.csv` is used for correctness auditing; the performance `comparison/runs.csv` is used for eigenvalue plots. Do not interchange them.

No duplicate Part C CPU scaling results, historical local performance measurements, or pre-generated figures are included. New experiments and analysis outputs should use fresh directories.

The GPU Lanczos implementation and completed results are linked by a matching 15-file source fingerprint. Its local audit is `python3 scripts/audit_gpu_lanczos.py`; its submission entry is `scripts/run_vanda_gpu_lanczos.pbs`. See `gpu_lanczos/README.md` for the separate timing and validation protocol.
