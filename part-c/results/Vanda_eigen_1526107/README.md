# Vanda Job 1526107: Power Iteration and Lanczos

Recorded site: Vanda. Node: IN-001. Job: `1526107.stdct-mgmt-02`. The performance provenance records FP64, MPICH 4.1.1, 4 MPI ranks, and 2 OpenMP threads per rank.

## Original evidence

| Package path | Purpose |
|---|---|
| `comparison/runs.csv` | 165 performance measurements |
| `comparison/summary.csv` | 33 configuration summaries, five repeats each |
| `comparison/provenance.json` | Identity, parameters, hashes, and timing definitions |
| `validation/summary.json` | Original overall PASS for 48 solver evaluations, `serial_only=false` |
| `validation/comparisons.csv` | Recovered 48 individual validation records |

All five original files retain their supplied bytes. The validation CSV was supplied with a duplicate filename suffix and is stored under the canonical name `comparisons.csv`. The historical top-level `IMPORT_RECORD.json` describes the original four-file import; `validation/IMPORT_RECORD.json` documents this later CSV addition. Neither is a new Vanda measurement.

Validation measurement ID: `6d1a7a375b8a4c698ec732b4c2c5bc03`. Performance measurement ID: `ede83c38f066417f96d4c9e57e1c3cc1`. The CSV matches the validation summary, not the performance stage.

## What validation did

Six graph cases cover zero edges, a single vertex, one edge, a path, a cycle, and a self-loop with an isolated vertex. Both algorithms run serially and with four MPI ranks. Independent dense residuals test the Ritz pairs against the constructed lazy operator. Endpoint eigenvalues are compared with 0.5 for zero-edge inputs and 1 for the other cases. One-step power/Lanczos results test consistency from a shared initial vector; small graphs also exercise empty MPI row partitions.

The recovered CSV supports numerical reconstruction of endpoint errors and serial one-step differences. Using the archived validator's criteria, all endpoint checks satisfy both eigenvalue error and dense residual below `1e-8`; the maximum endpoint residual is approximately `1.0180e-10`. Serial one-step eigenvalue differences are zero in the stored values and residual differences are at most approximately `1.1102e-16`, below `1e-12`.

From the project root:

```bash
python3 scripts/audit_eigen_validation.py
```

This audits existing data without MPI or a GPU. See [the detailed validation method and table](../../docs/VANDA_EIGEN.md#validation-evidence-and-local-audit). Vector dumps and solver-reported residuals are not present, so their normalization errors and dense-versus-reported residual differences cannot be reconstructed. The original suite PASS remains the evidence for those checks.

## Performance experiment

The three graph families are uniform, hub, and community, each with `N=100000`, input `E=800000`, and seed `20261029`. Both algorithms use the same `B=(I+S)/2` operator and initial-vector seed. Power budgets are 10/20/50/100/200/500; Lanczos budgets are 5/10/20/30/50; each configuration is repeated five times.

`solve_s` includes numerical solving, projected solving/reconstruction as applicable, and direct residual evaluation, excluding graph loading and MPI startup. Lanczos's final direct-residual SpMV is included in `spmv_count`. Failure to reach `1e-8` at a small budget is an accuracy observation, not a failed execution. The sampled budgets do not establish exact minimum time to tolerance.

This operator differs from directed PageRank with damping 0.85. These measurements do not provide CPU-core scaling curves or serial-to-parallel algorithm speedups.

## Corresponding code and retained evidence

The C++ sources match `comparison/provenance.json` / `source_files`, and `scripts/compare_eigen.py` matches its `workflow_sha256`. The PBS scripts retain the sockets-provider default and additional environment logging for reproduction. Later script logging is not presented as an original job record.

Remote graph inputs, PBS/probe logs, vector dumps, and binaries were not supplied. Their absence is distinct from the now-included per-evaluation CSV.

## Local performance plots

From the project root:

```bash
python3 -m pip install -r requirements-visualization.txt
python3 scripts/visualize_vanda.py eigen \
  --runs results/Vanda_eigen_1526107/comparison/runs.csv \
  --out figures/Vanda_eigen_1526107
```

This reads recorded performance measurements and does not execute C++ solvers. Use a fresh empty output directory. No figures were generated during this documentation update.
