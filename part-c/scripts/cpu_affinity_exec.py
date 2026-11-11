#!/usr/bin/env python3
"""Record each launched rank's actual Linux CPU mask, then exec the CPU solver."""
import json
import os
from pathlib import Path
import platform
import sys

prefix, *command = sys.argv[1:]
if not command or not hasattr(os, 'sched_getaffinity'):
    raise RuntimeError('A solver command and Linux CPU affinity support are required')
rank = next((os.environ[k] for k in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID']
             if k in os.environ), None)
if rank is None:
    raise RuntimeError('No supported MPI rank environment variable')
Path(f'{prefix}.rank{int(rank)}.json').write_text(json.dumps(dict(
    rank=int(rank), hostname=platform.node(), cpus=sorted(os.sched_getaffinity(0)),
    OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS'), command=command), indent=2) + '\n')
os.execvpe(command[0], command, os.environ)
