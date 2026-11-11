#!/usr/bin/env python3
"""Measure power/Lanczos on the same lazy operator; record actual execution provenance."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import time
import uuid
from process_logging import run_logged

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ['site', 'hostname', 'job_id', 'measurement_id', 'hardware_tag', 'source_sha256',
            'binary_sha256', 'precision', 'ranks', 'threads', 'execution_mode', 'mpi_implementation']
GROUP = [*IDENTITY, 'model', 'graph_sha256', 'N', 'input_E', 'seed', 'operator', 'algorithm', 'budget', 'residual_tolerance']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(str(r[k]) for k in GROUP), []).append(r)
    summary = []
    for _, items in sorted(groups.items()):
        if len({int(r['steps']) for r in items}) != 1:
            raise ValueError('Different actual step counts in one budget group; inspect raw measurements')
        s = {k:items[0][k] for k in GROUP}
        s.update(samples=len(items), actual_steps=int(items[0]['steps']), spmv_count=int(items[0]['spmv_count']),
                 solve_s_median=statistics.median(float(r['solve_s']) for r in items),
                 solve_s_min=min(float(r['solve_s']) for r in items), solve_s_max=max(float(r['solve_s']) for r in items),
                 residual_median=statistics.median(float(r['direct_residual']) for r in items),
                 all_reached_tol=all(float(r['direct_residual']) < float(r['residual_tolerance']) for r in items))
        summary.append(s)
    return summary


def plot_eigen(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = sorted({r['model'] for r in summary})
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.4*len(models)), squeeze=False, layout='constrained')
    for i, model in enumerate(models):
        for algorithm in ['power', 'lanczos']:
            items = sorted([r for r in summary if r['model'] == model and r['algorithm'] == algorithm], key=lambda r:int(r['budget']))
            if not items:
                continue
            residual = [max(float(r['residual_median']), 1e-17) for r in items]
            axes[i, 0].semilogy([int(r['spmv_count']) for r in items], residual, 'o-', label=algorithm)
            axes[i, 1].loglog([float(r['solve_s_median']) for r in items], residual, 'o-', label=algorithm)
        for ax in axes[i]:
            ax.axhline(float(summary[0]['residual_tolerance']), ls=':', c='gray')
            ax.set_ylabel(f'{model}: direct residual')
            ax.grid(alpha=.2)
            ax.legend()
        axes[i, 0].set_xlabel('SpMV calls, including final direct validation')
        axes[i, 1].set_xlabel('Complete numerical solve time (s)')
    fig.suptitle(f"Same B=(I+S)/2: power vs Lanczos | {summary[0]['site']} | N={summary[0]['N']} | {summary[0]['execution_mode']} p{summary[0]['ranks']}t{summary[0]['threads']} | job {summary[0]['job_id']}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf']:
        fig.savefig(out / ('comparison.' + extension), dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--site', choices=['local', 'Atlas', 'Vanda'], required=True)
    ap.add_argument('--hardware-tag', required=True)
    ap.add_argument('--bin-dir', default='bin')
    ap.add_argument('--work-dir')
    ap.add_argument('--cpu-prefix', '--launcher-prefix', dest='cpu_prefix', default=os.getenv('CPU_PREFIX', ''))
    ap.add_argument('--mpi-ranks', type=int, default=0, help='Positive value selects the actual MPI C++ binaries')
    ap.add_argument('--omp-threads', type=int, default=1)
    ap.add_argument('--mpi-implementation', default='none')
    ap.add_argument('--N', type=int, default=100000)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--power-budgets', default='10,20,50,100,200,500')
    ap.add_argument('--lanczos-budgets', default='5,10,20,30,50')
    ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    job = os.getenv('PBS_JOBID') or os.getenv('SLURM_JOB_ID') or 'none'
    if a.site != 'local' and job == 'none':
        ap.error('Cluster measurements require a scheduler allocation; do not run on a login node')
    if job != 'none' and a.site == 'local':
        ap.error('Use the actual cluster site for a scheduler job')
    if a.N < 20 or a.repeats < 1 or not math.isfinite(a.tol) or a.tol <= 0:
        ap.error('N>=20 and positive repeats/tolerance required')
    choices = [('power', list(map(int, a.power_budgets.split(',')))), ('lanczos', list(map(int, a.lanczos_budgets.split(','))))]
    for _, values in choices:
        if not values or min(values) < 1 or len(values) != len(set(values)):
            ap.error('Budgets must be distinct positive integers')
    if a.mpi_ranks < 0 or a.omp_threads < 1:
        ap.error('Nonnegative MPI ranks and positive thread count required')
    if a.mpi_ranks and (not a.cpu_prefix or a.mpi_implementation == 'none'):
        ap.error('MPI measurement requires an explicit launcher prefix and MPI implementation')
    if not a.mpi_ranks and a.omp_threads != 1:
        ap.error('Serial binaries require one thread')
    expected_ranks = a.mpi_ranks or 1
    expected_threads = a.omp_threads
    suffix = '_mpi' if a.mpi_ranks else ''
    worker_env = os.environ.copy()
    worker_env.update(OMP_NUM_THREADS=str(expected_threads), OMP_DYNAMIC='FALSE',
                      OMP_PROC_BIND='close', OMP_PLACES='cores')
    bins = (ROOT / a.bin_dir).resolve()
    executables = ['make_graph', 'power_symmetric'+suffix, 'lanczos'+suffix]
    for name in executables:
        if not os.access(bins / name, os.X_OK):
            raise FileNotFoundError(bins / name)
    out = (ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a fresh empty --out directory')
    work = (ROOT / a.work_dir).resolve() if a.work_dir else out / 'inputs'
    work.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT / 'src').glob('*')) if p.is_file()}
    binaries = {name:sha(bins / name) for name in executables}
    identity = dict(site=a.site, hostname=platform.node(), job_id=job, measurement_id=uuid.uuid4().hex,
                    hardware_tag=a.hardware_tag, source_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                    binary_sha256=hashlib.sha256(json.dumps(binaries, sort_keys=True).encode()).hexdigest(), precision='fp64', ranks=expected_ranks, threads=expected_threads,
                    execution_mode='MPI+OpenMP' if a.mpi_ranks else 'serial', mpi_implementation=a.mpi_implementation)
    meta = dict(identity, status='RUNNING', configuration=vars(a), source_files=sources, binaries=binaries,
                workflow_sha256=sha(__file__), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                parent_cpu_mask=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                OMP_environment={k:v for k,v in os.environ.items() if k.startswith('OMP_')}, graphs=[], completed_runs=0,
                operator='B=(I+S)/2; identical input and initial seed',
                time_definition='solve_s includes projected solve, reconstruction and direct residual; excludes graph loading and MPI launcher overhead',
                spmv_definition='steps + 1 final direct residual call for Lanczos; steps for power',
                criterion='Direct residual at tested budgets; not exact minimum time-to-tolerance')
    def save_meta():
        (out / 'provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    def invoke(command, label, env=None):
        started = time.perf_counter()
        p = run_logged(command, out / label, env=env, timeout=1800)
        elapsed = time.perf_counter() - started
        return p.stdout, elapsed
    save_meta()
    try:
        graphs = []
        for model in ['uniform', 'hub', 'community']:
            path = work / f'{model}_n{a.N}_s{a.seed}.txt'
            if path.exists():
                raise FileExistsError(f'Use a fresh --work-dir: {path}')
            invoke([str(bins / 'make_graph'), model, str(a.N), str(a.N*8), str(a.seed), str(path)], 'generate_' + model)
            g = dict(model=model, path=str(path), sha256=sha(path), N=a.N, input_E=a.N*8, seed=a.seed)
            graphs.append(g)
        meta['graphs'] = graphs
        save_meta()
        jobs = [(g, algorithm, budget, rep) for g in graphs for rep in range(1, a.repeats+1)
                for algorithm, values in choices for budget in values]
        random.Random(20261103).shuffle(jobs)
        rows = []
        with (out / 'runs.csv').open('w', newline='') as stream:
            writer = None
            for i, (g, algorithm, budget, rep) in enumerate(jobs, 1):
                exe = ('power_symmetric' if algorithm == 'power' else 'lanczos') + suffix
                command = [*shlex.split(a.cpu_prefix), str(bins / exe), g['path'], str(budget), '--seed', str(a.seed), '--operator', 'lazy']
                if algorithm == 'lanczos':
                    command += ['--reorth', 'full']
                label = f'{i:04d}_{g["model"]}_{algorithm}_b{budget}_r{rep}'
                stdout, wall = invoke(command, label, env=worker_env)
                m = dict(line.split(': ', 1) for line in stdout.splitlines() if ': ' in line)
                if m['operator'] != 'lazy' or int(m['ranks']) != expected_ranks or int(m['threads']) != expected_threads:
                    raise ValueError('Actual operator/rank/thread count differs from the requested numerical comparison')
                row = dict(identity, run_id=label, model=g['model'], graph_sha256=g['sha256'], N=a.N, input_E=a.N*8,
                           seed=a.seed, algorithm=algorithm, operator='lazy', budget=budget, residual_tolerance=a.tol,
                           steps=int(m['steps']), spmv_count=int(m['steps'])+(1 if algorithm == 'lanczos' else 0),
                           repeat=rep, solve_s=float(m['solve_s']), kernel_s=float(m['iteration_s']),
                           lambda_ritz=float(m['lambda_max_ritz']), direct_residual=float(m['ritz_residual']), launcher_wall_s=wall)
                if any(not math.isfinite(row[k]) for k in ['solve_s', 'kernel_s', 'lambda_ritz', 'direct_residual']):
                    raise ValueError('Nonfinite solver metric')
                if row['solve_s'] <= 0 or row['kernel_s'] < 0 or row['direct_residual'] < 0:
                    raise ValueError('Invalid solver metric')
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                meta['completed_runs'] = len(rows)
                save_meta()
                print(f'{i}/{len(jobs)} {label} solve_s={row["solve_s"]:.9g} residual={row["direct_residual"]:.6g}', flush=True)
        summary = summarize_rows(rows)
        write_csv(out / 'summary.csv', summary)
        meta.update(status='COMPLETE', configurations=len(summary), finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save_meta()
    except Exception as exc:
        meta.update(status='FAILED', error=str(exc))
        save_meta()
        raise
    if not a.no_plot:
        plot_eigen(summary, out)
    print(f'EIGEN_COMPLETE: {len(rows)} actual runs, {len(summary)} configurations. Results: {out}')


if __name__ == '__main__':
    main()
