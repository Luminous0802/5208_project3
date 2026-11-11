#!/usr/bin/env python3
"""Input/binding safeguards only; no GPU or cluster measurements."""
import json
from pathlib import Path
import platform
from benchmark_gpu import ROOT, check_affinity, load_graphs, sha


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected rejection')


def main():
    work = ROOT / 'work/gpu_inputs_unit'
    work.mkdir(parents=True, exist_ok=True)
    graph = work / 'graph.txt'
    graph.write_text('2 1\n0 1\n')
    item = dict(graph_id='unit', path=str(graph), N=2, E=1, sha256=sha(graph))
    manifest = work / 'manifest.json'
    manifest.write_text(json.dumps([item]))
    assert load_graphs(manifest)[0]['sha256'] == item['sha256']
    graph.write_text('2 1\n1 0\n')  # Same N/E, different graph: must reject.
    rejected(lambda: load_graphs(manifest))
    manifest.write_text(json.dumps([dict(item, sha256=sha(graph), E=2)]))
    rejected(lambda: load_graphs(manifest))
    prefix = work / 'affinity'

    def rank(r, cpus, host=None):
        Path(f'{prefix}.rank{r}.json').write_text(json.dumps(dict(rank=r, cpus=cpus, hostname=host or platform.node())))

    rank(0, [2, 3]); rank(1, [4, 5])
    check_affinity(prefix, 2, 2)
    rank(1, [3, 4]); rejected(lambda: check_affinity(prefix, 2, 2))
    rank(1, [4]); rejected(lambda: check_affinity(prefix, 2, 2))
    rank(1, [4, 5], 'OTHER_UNIT_HOST'); rejected(lambda: check_affinity(prefix, 2, 2))
    print('PASS graph hash/header and overlapping/undersized/off-node CPU-mask safeguards; no GPU execution.')


if __name__ == '__main__':
    main()
