# Locally Generated Figures

No pre-generated PNG/PDF figures or example curves are included. Run `scripts/visualize_vanda.py` from the project root to read recorded Vanda performance `runs.csv` files and generate summaries and figures in a fresh output directory.

Measurements were made on Vanda. `plot_provenance.json` records where the analysis was performed; locally generated plots do not represent new local timing measurements. The validation `comparisons.csv` is audited separately and is not a performance-plot input.

GPU Lanczos has a separate plotter, `gpu_lanczos/scripts/analyze_gpu_lanczos.py`, which reads `results/Vanda_gpu_lanczos_1536096/runs.csv`. Its command is documented in the module README; it does not use `visualize_vanda.py`.
