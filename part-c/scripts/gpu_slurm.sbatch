#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-gpu-c-%j.out
# Generic Slurm GPU request, NOT verified site partition/account settings.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HPC_SITE:?Set Atlas or Vanda}"
: "${HARDWARE_TAG:?Set actual CPU/GPU node type}"
: "${GPU_MANIFEST:?Set graph manifest path}"
if [[ -n "${HPC_GPU_MODULES:-}" ]]; then read -r -a mods <<< "$HPC_GPU_MODULES"; module load "${mods[@]}"; fi
result_dir="$PWD/results/${HPC_SITE}_gpu_${SLURM_JOB_ID}"
mkdir -p "$result_dir"
export BIN_DIR="$PWD/work/gpu_build_${SLURM_JOB_ID}/bin"
export VALIDATION_OUT="$result_dir/validation"
bash scripts/gpu_preflight.sh > "$result_dir/preflight.log" 2>&1
extra=()
if [[ -n "${GPU_HYBRID_CONFIGS:-}" ]]; then
  : "${GPU_CPU_CORES:?Set actual allocated cores}"
  : "${MPIEXEC_ARGS:?Set mapping compatible with the actual MPI and Slurm allocation}"
  extra=(--hybrid-configs "$GPU_HYBRID_CONFIGS" --cpu-cores "$GPU_CPU_CORES")
fi
python3 scripts/benchmark_gpu.py "$GPU_MANIFEST" --out "$result_dir" --bin-dir "$BIN_DIR" \
  --site "$HPC_SITE" --hardware-tag "$HARDWARE_TAG" --repeats 5 --K 50 --block-sizes "${BLOCK_SIZES:-256}" "${extra[@]}"
python3 scripts/analyze_gpu.py "$result_dir/runs.csv" --out "$result_dir/summary.csv" --plot-dir "$result_dir/plots"
