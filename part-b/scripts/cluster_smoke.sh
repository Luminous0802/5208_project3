#!/bin/bash
# Run inside an allocation of at least 8 CPU cores on Atlas/Vanda after loading the MPI/compiler modules.
set -euo pipefail
export OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
make -j 4
# Set MPIEXEC and MPIEXEC_ARGS for the selected cluster/site before invoking this script.
python3 scripts/validate.py --out results/cluster_validation
python3 scripts/prepare_experiments.py --out work/smoke_graphs --N 120 \
  --sizes 120 --degrees 4,8,16 --seeds 20261029 --experiments E1,E2,E3,E4 --generate
python3 scripts/benchmark.py work/smoke_graphs/manifest.json --configs scripts/configs_smoke.json \
  --max-cores 4 --repeats 1 --site "${SITE:-local}" --out "results/smoke_${SITE:-local}_$(date +%Y%m%d_%H%M%S)"
