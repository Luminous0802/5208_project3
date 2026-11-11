# Vanda GPU Lanczos Job 1536096

This directory retains three original downloaded CSV files from job `1536096.stdct-mgmt-02`, node GN-A40-003, with one NVIDIA A40:

| File | Contents |
|---|---|
| `runs.csv` | 300 performance measurements |
| `summary.csv` | 60 configurations, five repeats each |
| `validation/comparisons.csv` | 48 small-graph validation records |

All three files retain their supplied bytes. This archive stores the downloaded records directly in the job folder; a fresh PBS workflow writes its records under `experiment/`.

The imported source set matches the complete `source_sha256` fingerprint in the timing CSV. The source import and its historical pre-run checks are documented in `../../docs/GPU_LANCZOS_SOURCE_IMPORT.json`. `IMPORT_RECORD.json` records file hashes and integration checks rather than new measurements.

Four variants compare serial CPU, 2 MPI x 2 OpenMP CPU, one-process GPU, and two-process shared-GPU execution. Both GPU variants use the same physical A40. Three graph families use N=100000, input E=800000, seed 20261029, FP64, the lazy symmetric operator, and full reorthogonalization. Budgets are 5/10/20/30/50.

At the selected common passing budgets 20/20/30, one-process GPU solve ratios relative to the same-node 2x2 CPU baseline are 4.33/3.08/4.66 for Uniform/Hub/Community. Broader application ratios span only 0.87--1.07. These records are separate from the earlier CPU power/Lanczos comparison and do not demonstrate multi-GPU scaling.

The 48 validation rows report PASS, with full-budget dense residual at most approximately 7.8945e-16 and one-step vector L1 difference at most approximately 4.1980e-16. These diagnostics are retained; original vectors, runtime provenance JSON, binaries, allocation/binding logs, and other original raw logs were not supplied.

From the project root:

```bash
python3 scripts/audit_gpu_lanczos.py
```

For implementation, validation thresholds, timing scopes, reproduction, and plotting commands, see [the module README](../../gpu_lanczos/README.md).
