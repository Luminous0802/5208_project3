#!/bin/bash
# Run inside an allocated NVIDIA GPU job, after loading actual compiler/CUDA modules.
set -euo pipefail
command -v nvcc
command -v nvidia-smi
nvidia-smi
nvcc --version
bin_dir=${BIN_DIR:-bin}
validation_out=${VALIDATION_OUT:-results/gpu_validation}
make BIN="$bin_dir" CUDA_ARCH="${CUDA_ARCH:-}" gpu
make BIN="$bin_dir" "$bin_dir/pagerank_reference"
if [[ -n "${GPU_HYBRID_CONFIGS:-}" ]]; then
  # Load the actual MPI/compiler modules first; CPU and CUDA builds are independent.
  make BIN="$bin_dir" "$bin_dir/pagerank_hybrid"
fi
python3 scripts/validate_gpu.py --bin-dir "$bin_dir" --out "$validation_out"
if command -v compute-sanitizer >/dev/null 2>&1; then
  compute-sanitizer --error-exitcode=9 --tool memcheck "$bin_dir/pagerank_cuda" tests/check_graph.txt 50 --mapping thread --validate
  compute-sanitizer --error-exitcode=9 --tool racecheck "$bin_dir/pagerank_cuda" tests/check_graph.txt 50 --mapping warp --validate
else
  echo 'Compute Sanitizer unavailable; memory/race checks have NOT run.'
fi
