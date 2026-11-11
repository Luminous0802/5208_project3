#!/bin/bash
# Edit only the environment values below for the actual account/site.
# Vanda module confirmed by A: openmpi/gcc/64/4.1.8
# Atlas module/version has NOT been supplied; do not assume Vanda's module works there.
if [[ -n "${HPC_MODULES:-}" ]]; then
    read -r -a requested_modules <<< "$HPC_MODULES"
    module load "${requested_modules[@]}"
fi
: "${HPC_SITE:?Set HPC_SITE=Atlas or HPC_SITE=Vanda}"
: "${HARDWARE_TAG:?Set a CPU model / node-type tag consistent across comparable jobs}"
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export MPIEXEC="${MPIEXEC:-mpirun}"
