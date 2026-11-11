#!/usr/bin/env python3
"""Rebuild/reuse the received B graphs, requiring their original SHA256 values."""
import argparse
import json
from pathlib import Path
import subprocess
from benchmark_gpu import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', default='configs/part_b_graphs.json')
    ap.add_argument('--suites', default='B3,B4')
    ap.add_argument('--seeds', default='20261029')
    ap.add_argument('--graph-dir', default='work/part_b_gpu_graphs')
    ap.add_argument('--out', default='work/part_b_gpu_manifest.json')
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--bin-dir', default='bin')
    a = ap.parse_args()
    suites = set(a.suites.split(','))
    seeds = set(map(int, a.seeds.split(',')))
    graphs = [dict(g) for g in json.loads((ROOT / a.catalog).read_text())
              if set(g['part_b_suites']) & suites and g['seed'] in seeds]
    if not graphs:
        raise ValueError('Empty graph selection')
    for g in graphs:
        g['path'] = str(Path(a.graph_dir) / (g['graph_id'] + '.txt'))
        if sha(ROOT / 'src/make_graph.cpp') != g['generator_sha256']:
            raise ValueError('Generator differs from received B source')
        if a.generate:
            p = ROOT / g['path']
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                subprocess.run([str((ROOT / a.bin_dir / 'make_graph').resolve()), g['model'],
                                str(g['N']), str(g['E']), str(g['seed']), str(p), *g['options']], check=True)
            if sha(p) != g['sha256']:
                raise ValueError(f'Graph differs from B: {p}; no manifest written')
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graphs, indent=2) + '\n')
    print(f'{len(graphs)} graphs: {out}; ' + ('B SHA256 values verified' if a.generate else 'PLAN ONLY'))


if __name__ == '__main__':
    main()
