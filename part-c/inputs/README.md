# Additional Downloaded Inputs

This directory has no preloaded measurement files. Store additional original CSV files or metadata downloaded from Vanda in separate subdirectories such as `gpu_JOBID/`, `part_a_JOBID/`, or `part_b_JOBID/` to avoid overwriting files named `runs.csv`. Keep original evidence unchanged and write derived summaries/figures to a fresh directory under `figures/`.

The included power/Lanczos measurements are already under `../results/Vanda_eigen_1526107/comparison/`. Use their `runs.csv` directly for plotting. The recovered correctness records are under `../results/Vanda_eigen_1526107/validation/comparisons.csv` and are audited separately.

The included GPU/12-core CPU measurements are under `../results/Vanda_gpu_1520880/`. Their `runs.csv` can also be plotted directly without copying it here.

The GPU Lanczos measurements are retained under `../results/Vanda_gpu_lanczos_1536096/`, with their validation records in `validation/comparisons.csv`. Use the separate module analysis command in `../gpu_lanczos/README.md`.
