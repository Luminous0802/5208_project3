#!/usr/bin/env python3
"""Index only explicitly supplied Vanda summaries; never load historical defaults."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part-a-summary', '--cpu-summary', dest='part_a_summary',
                    action='append', type=Path, default=[],
                    help='Existing Vanda summary supplied by teammate A; no new scaling experiment')
    for kind in ['gpu', 'part-b', 'eigen', 'gpu-lanczos']:
        ap.add_argument(f'--{kind}-summary', action='append', type=Path, default=[])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    groups = [('A_pagerank_scaling', a.part_a_summary), ('GPU_new_comparison', a.gpu_summary),
              ('B_hybrid_structure', a.part_b_summary), ('symmetric_eigen_comparison', a.eigen_summary),
              ('GPU_Lanczos_comparison', a.gpu_lanczos_summary)]
    if not any(paths for _, paths in groups):
        ap.error('Supply at least one real Vanda summary explicitly')
    index, tables, seen = [], [], set()
    for kind, paths in groups:
        for supplied in paths:
            path = supplied.expanduser().resolve()
            if path in seen:
                raise ValueError(f'Duplicate summary: {path}')
            seen.add(path)
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f'Empty summary: {path}')
            for row in rows:
                if row.get('site') != 'Vanda':
                    raise ValueError(f'{path}: missing site=Vanda; local/unknown data is not integrated')
                job = row.get('job_id') or row.get('job_ids') or ''
                if not job or any(not item or not item[0].isdigit() for item in job.split(';')):
                    raise ValueError(f'{path}: missing Vanda PBS job ID')
                tables.append(dict(section=kind, record=row))
            index.append(dict(kind=kind, path=str(path), rows=len(rows),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    out = a.out.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('Choose a fresh, empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    part_a = [x['record'] for x in tables if x['section'] == 'A_pagerank_scaling']
    part_b = [x['record'] for x in tables if x['section'] == 'B_hybrid_structure']
    eigen = [x['record'] for x in tables if x['section'] == 'symmetric_eigen_comparison']
    gpu_lanczos = [x['record'] for x in tables if x['section'] == 'GPU_Lanczos_comparison']
    gpu = [x['record'] for x in tables if x['section'] == 'GPU_new_comparison']
    coverage = dict(measurement_site='Vanda', summaries=len(index), summary_rows=len(tables),
                    GPU_summary_supplied=bool(gpu),
                    GPU_validated_rows_supplied=any(r.get('backend') == 'cuda' and r.get('precision') == 'fp64'
                        and r.get('gpu_name') and r.get('validation_L1_max', '') != ''
                        and 0 <= float(r['validation_L1_max']) < 1e-8 for r in gpu),
                    part_A_summary_supplied=bool(part_a), part_B_summary_supplied=bool(part_b),
                    eigen_summary_supplied=bool(eigen),
                    GPU_Lanczos_summary_supplied=bool(gpu_lanczos),
                    rules=['Inputs were supplied explicitly; bundled results are never loaded implicitly',
                           'Measurement metadata is checked, not independently authenticated',
                           'Keep original Vanda job logs, metadata and input hashes with the final evidence',
                           'No cross-site, cross-hardware or cross-algorithm speedups are calculated'])
    for name, value in [('source_index.json', index), ('tables.json', tables), ('coverage.json', coverage)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    print(f'Indexed {len(index)} user-supplied Vanda summaries: {out}')


if __name__ == '__main__':
    main()
