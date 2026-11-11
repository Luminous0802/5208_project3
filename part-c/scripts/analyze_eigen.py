#!/usr/bin/env python3
"""Summarize and plot an explicitly supplied Vanda eigen runs.csv; no solver execution."""
import argparse
import csv
import math
from pathlib import Path
from compare_eigen import GROUP, summarize_rows, write_csv, plot_eigen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plot-dir', type=Path, required=True)
    a = ap.parse_args()
    with a.runs.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty input')
    seen = set()
    for row in rows:
        if row['site'] != 'Vanda' or row['operator'] != 'lazy' or row['precision'] != 'fp64':
            raise ValueError('Require actual Vanda FP64 measurements of the same lazy operator')
        if row['algorithm'] not in ['power', 'lanczos'] or int(row['ranks']) < 1 or int(row['threads']) < 1:
            raise ValueError('Invalid algorithm/process/thread count')
        if row['execution_mode'] == 'serial' and (row['ranks'] != '1' or row['threads'] != '1'):
            raise ValueError('Serial measurement has a parallel layout')
        if row['execution_mode'] not in ['serial', 'MPI+OpenMP']:
            raise ValueError('Unknown execution mode')
        if row['execution_mode'] == 'MPI+OpenMP' and row['mpi_implementation'] == 'none':
            raise ValueError('MPI implementation was not recorded')
        if not row['job_id'] or not row['job_id'][0].isdigit():
            raise ValueError('Missing PBS job ID')
        for key in ['solve_s', 'kernel_s', 'direct_residual', 'residual_tolerance']:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError(f'Invalid {key}')
        if float(row['solve_s']) <= 0 or float(row['residual_tolerance']) <= 0:
            raise ValueError('Nonpositive solve time or tolerance')
        identity = (row['measurement_id'], row['run_id'])
        if identity in seen:
            raise ValueError('Duplicate run')
        seen.add(identity)
    common = ['site', 'hostname', 'job_id', 'measurement_id', 'hardware_tag', 'source_sha256', 'binary_sha256', 'seed', 'N', 'input_E', 'residual_tolerance', 'ranks', 'threads', 'execution_mode', 'mpi_implementation']
    if len({tuple(r[k] for k in common) for r in rows}) != 1:
        raise ValueError('Analyze one compatible measurement at a time')
    summary = summarize_rows(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(a.out, summary)
    plot_eigen(summary, a.plot_dir)
    print(f'EIGEN SUMMARY: {len(summary)} measured configurations -> {a.out}')


if __name__ == '__main__':
    main()
