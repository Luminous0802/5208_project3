# Figure guide and recommended report placement

All images are available as high-resolution PNG and vector PDF. All timings represent 50 iterations unless otherwise stated. Logarithmic time axes are marked explicitly.

| Figure | Topic | Recommended section |
|---|---|---|
| 01 | OpenMP speedup vs ideal | OpenMP scaling |
| 02 | OpenMP efficiency | OpenMP scaling |
| 03 | Fixed-core MPI/OMP layouts, with min–max | Hybrid design |
| 04 | Compute/Allreduce/Allgatherv maxima | Communication analysis |
| 05 | Graph structure V2 vs V3 | Graph structure |
| 06 | Speedup heatmap: partition × schedule | Load balancing |
| 07 | Static vs dynamic for uniform/hub | Thread scheduling |
| 08 | Community vertex ordering | Graph ordering |
| 09 | B4 paired speedup with seed range | Robustness |
| 10 | Edge imbalance across MPI ranks | Partition analysis |
| 11 | Hub iteration and phase maxima | Bottleneck discussion |
| 12 | Iteration vs end-to-end | Measurement limitations |

Suggested main-text figures: 01, 03, 04, 05, 06, 09, 10. Others are suitable for the appendix. Do not present all twelve if page limits make the report harder to read.

**Statistics:** Each configuration is summarized by the median of five runs. Error bars on Figure 03 show the observed minimum and maximum across those five runs. Figure 09 shows seed-wise speedups and their observed min–max across three graph seeds; it is not a 95% confidence interval. All figures use the same solver source hash per experiment and exclude failed job IDs.
