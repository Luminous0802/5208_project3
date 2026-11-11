#!/usr/bin/env python3
"""Run stage-logged MPI/OpenMP startup probes inside the allocated PBS job."""
import argparse
import os
from pathlib import Path
import shlex
from process_logging import run_logged


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--timeout',type=int,default=45)
    a=ap.parse_args()
    if not os.getenv('PBS_JOBID'):
        ap.error('Run inside an allocated PBS job')
    if a.timeout<=0:ap.error('Positive timeout required')
    launcher=shlex.split(os.environ['MPIEXEC'])+shlex.split(os.environ['MPIEXEC_ARGS'])
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_PROC_BIND='FALSE')
    for ranks in [1,4]:
        print(f'MPI_PROBE_START ranks={ranks} threads=2',flush=True)
        result=run_logged([*launcher,'-n',str(ranks),str(a.binary.resolve())],
                          a.out/f'p{ranks}_t2',env=env,timeout=a.timeout)
        expected=os.getenv('EIGEN_EXPECTED_MPI')
        if expected and expected not in result.stderr:
            raise RuntimeError(f'Actual MPI library does not identify as {expected}; inspect probe stderr')
        print(result.stdout,flush=True)
    print('MPI_EIGEN_STARTUP_PASS',flush=True)


if __name__=='__main__':main()
