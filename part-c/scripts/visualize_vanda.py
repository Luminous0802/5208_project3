#!/usr/bin/env python3
"""Read user-supplied Vanda runs.csv and produce summaries/figures only.

This entry point never compiles code, generates graphs, launches a solver or
submits jobs. Plotting can run on a local computer; measurement site stays Vanda.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_input(path, kind):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        required = {'site', 'job_id', 'measurement_id', 'run_id', 'hardware_tag', 'source_sha256', 'graph_sha256'}
        if kind == 'gpu':
            required |= {'iteration_s', 'K', 'backend', 'precision', 'hostname', 'build_tag'}
        else:
            required |= {'solve_s', 'kernel_s', 'direct_residual', 'residual_tolerance', 'hostname', 'binary_sha256', 'precision',
                         'ranks', 'threads', 'operator', 'algorithm', 'model', 'N', 'input_E', 'seed', 'budget', 'steps', 'spmv_count', 'repeat', 'execution_mode', 'mpi_implementation'}
        if not required <= fields:
            raise ValueError(f'Use original {kind} runs.csv; missing columns: {sorted(required - fields)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Input CSV has no measurements')
    identities = set()
    for number, row in enumerate(rows, 2):
        if row['site'] != 'Vanda':
            raise ValueError(f'Line {number}: only site=Vanda is accepted; local/unknown data is not plotted')
        job = row['job_id'].strip()
        if not job or job.lower() in {'none', 'local', 'unknown'} or not job[0].isdigit():
            raise ValueError(f'Line {number}: missing Vanda PBS job ID')
        for key in ['measurement_id', 'hardware_tag', 'source_sha256', 'graph_sha256']:
            if not row[key].strip():
                raise ValueError(f'Line {number}: missing {key}')
        elapsed = float(row['solve_s' if kind == 'eigen' else 'iteration_s'])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f'Line {number}: invalid iteration time')
        identity = (job, row['measurement_id'], row['run_id'])
        if identity in identities:
            raise ValueError(f'Line {number}: duplicate run')
        identities.add(identity)
        if kind in ['gpu', 'eigen'] and row['precision'] != 'fp64':
            raise ValueError(f'Line {number}: expected FP64 measurements')
    # One measurement per call avoids treating unrelated jobs as repetitions.
    common = ['job_id', 'measurement_id', 'site', 'hardware_tag', 'source_sha256']
    if kind == 'gpu':
        common += ['K', 'hostname', 'build_tag']
    else:
        common += ['hostname', 'binary_sha256', 'seed', 'N', 'input_E', 'residual_tolerance', 'ranks', 'threads', 'execution_mode', 'mpi_implementation']
    if len({tuple(row[k] for k in common) for row in rows}) != 1:
        raise ValueError('Supply one compatible measurement/job per invocation')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('kind', choices=['gpu', 'eigen'], help='gpu: CPU/GPU comparison; eigen: power/Lanczos')
    ap.add_argument('--runs', type=Path, required=True, help='Original runs.csv downloaded from Vanda')
    ap.add_argument('--out', type=Path, required=True, help='Fresh local output directory')
    a = ap.parse_args()
    source = a.runs.expanduser().resolve()
    rows = check_input(source, a.kind)
    try:
        import matplotlib  # noqa: F401; check dependency before creating output
    except ImportError as exc:
        raise SystemExit('Install plotting dependency: python3 -m pip install -r requirements-visualization.txt') from exc
    out = a.out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Choose a fresh, empty --out directory; existing results are not overwritten')
    out.mkdir(parents=True, exist_ok=True)
    summary, figures = out / 'summary.csv', out / 'plots'
    if a.kind == 'gpu':
        commands = [[sys.executable, str(ROOT / 'scripts/analyze_gpu.py'), str(source),
                     '--out', str(summary), '--plot-dir', str(figures)]]
    else:
        commands = [[sys.executable, str(ROOT / 'scripts/analyze_eigen.py'), str(source),
                     '--out', str(summary), '--plot-dir', str(figures)]]
    provenance = dict(status='ANALYSIS_RUNNING', measurement_site='Vanda',
                      analysis_host=platform.node(), analysis_platform=platform.platform(),
                      analysis_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      input_runs=str(source), input_sha256=sha(source), input_rows=len(rows),
                      job_id=rows[0]['job_id'], measurement_id=rows[0]['measurement_id'],
                      commands=commands, workflow_sha256={str(Path(c[1]).relative_to(ROOT)):sha(Path(c[1])) for c in commands},
                      note='Analysis/plotting only. CSV metadata is checked, not independent authentication of remote execution. No timings are generated or filled in.')
    if a.kind == 'eigen':
        provenance['workflow_sha256']['scripts/compare_eigen.py'] = sha(ROOT / 'scripts/compare_eigen.py')
    metadata_path = out / 'plot_provenance.json'

    def save():
        metadata_path.write_text(json.dumps(provenance, indent=2) + '\n')

    save()
    try:
        for command in commands:
            subprocess.run(command, check=True)
        if sha(source) != provenance['input_sha256']:
            raise RuntimeError('Input CSV changed during analysis; discard this output and retry')
        provenance.update(status='ANALYSIS_COMPLETE', summary_sha256=sha(summary),
                          figures={p.name:sha(p) for p in sorted(figures.iterdir()) if p.is_file()})
        save()
    except Exception as exc:
        provenance.update(status='ANALYSIS_FAILED', error=str(exc))
        save()
        raise
    print(f'Vanda measurements: {len(rows)} rows from job {rows[0]["job_id"]}')
    print(f'Local analysis output: {out}')


if __name__ == '__main__':
    main()
