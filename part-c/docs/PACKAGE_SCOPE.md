# Package Scope

This is the final Part C source-and-evidence package. It includes the implementations, job scripts, small test inputs, graph specifications, and ten original Vanda result files: two PageRank GPU performance CSV files, five CPU eigenvalue performance/validation CSV/JSON files, and three GPU Lanczos CSV files. The recovered 48-row validation CSV is included as `results/Vanda_eigen_1526107/validation/comparisons.csv`.

The package excludes duplicate Part C CPU strong/weak scaling workflows, A/B result archives, historical local performance results, generated figures, report PDFs, compiled binaries, work directories, and caches. Group-wide scaling analysis uses Member A's measurements. Shared baseline sources remain available for comparison and compilation.

`configs/part_b_graphs.json` stores graph sizes, seeds, generator options, and hashes, rather than performance measurements. The GPU preparation tool reads this input specification. The integration tool accepts explicitly named A/B/GPU/eigenvalue summaries. The plotting entry point requires Vanda identity fields; these fields alone are not independent proof of server execution.

The numerical C++/CUDA sources retain their original contents. Source and measurement-driver hashes match the supplied performance provenance. The recovered validation CSV retains its bytes after filename normalization. The archived validation driver supplies criteria for a local evidence audit; its own hash was not recorded in the original validation summary.

The MPICH entry point requests a single allocated node, probes real MPI execution, performs serial/MPI small-graph validation, and samples the 4-rank x 2-thread layout. The sockets-provider request and binding options are configuration records, not independent proof of a particular message transport or exclusive CPU placement.

The current English documentation describes the completed measurements and the recovered validation details. Local checks during this update analyze existing files; they do not run MPI/CUDA solvers, perform new timing experiments, or generate figures. Historical import records are preserved, and the added CSV has a separate import record. `CHECKSUMS.json` describes the current package contents.

## GPU Lanczos integration

The supplied `gpu-lanczos-vanda-experiment.zip` is merged as `gpu_lanczos/` plus `scripts/run_vanda_gpu_lanczos.pbs`. Existing shared source/helper files are byte-identical and are retained. The new module and all 15 files in its recorded source fingerprint remain unchanged; its Chinese pre-run README is replaced by English completed-experiment documentation, which is outside that fingerprint. The root Makefile adds an optional `gpu-lanczos` target without changing existing targets.

The source fingerprint matches job 1536096's `source_sha256`. Three original result CSVs are retained under `results/Vanda_gpu_lanczos_1536096/`. Historical static-check metadata from before the run is preserved in `GPU_LANCZOS_SOURCE_IMPORT.json`; it is not presented as a new cluster run or substituted for the completed CSV evidence. Source equality does not independently verify missing binary files, original vector dumps, allocation logs, or affinity logs.

The integration checks Python/Shell syntax, build dependencies, source hashes, numerical CSV summaries, validation diagnostics, links, English-only text, and archive integrity. It does not launch MPI/CUDA solvers, perform new benchmarks, or generate plots.
