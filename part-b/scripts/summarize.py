#!/usr/bin/env python3
"""Group repeated timings by graph/configuration; never treat different graph seeds as repeats."""
import argparse,csv,pathlib,statistics
METRICS=['iteration_s','compute_s','allreduce_s','allgatherv_s','load_s','partition_s','end_to_end_s','launcher_wall_s','edge_imbalance','rank_sum','residual']
KEYS=['site','graph_id','graph_sha256','seed','N','E','model','version','partition','schedule','chunk','ranks','threads','cores','K','source_sha256']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');ap.add_argument('--out',required=True);a=ap.parse_args()
    groups={}
    for path in a.runs:
        for row in csv.DictReader(open(path)):
            key=tuple(row[k] for k in KEYS);groups.setdefault(key,[]).append(row)
    summaries=[]
    for key,rows in groups.items():
        result=dict(zip(KEYS,key));result['samples']=len(rows)
        for m in METRICS:
            values=[float(r[m]) for r in rows if r[m]!='']
            for suffix,func in [('median',statistics.median),('min',min),('max',max)]:
                result[m+'_'+suffix]=func(values) if values else ''
        summaries.append(result)
    # Site, exact graph bytes, iteration protocol and source snapshot must match.
    baseline={}
    def match(s):return tuple(s[k] for k in ['site','graph_sha256','K','source_sha256'])
    for s in summaries:
        if s['version']=='V0':baseline[match(s)]=s['iteration_s_median']
    for s in summaries:
        base=baseline.get(match(s));runtime=s['iteration_s_median']
        speed=base/runtime if base is not None and runtime>0 else ''
        s['speedup_serial']=speed;s['efficiency']=speed/int(s['cores']) if speed!='' else ''
        s['comparison_note']='matched V0 median' if base is not None else 'no matching V0 baseline'
    path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[*KEYS,'samples',*[m+'_'+s for m in METRICS for s in ['median','min','max']],
            'speedup_serial','efficiency','comparison_note']);w.writeheader();w.writerows(summaries)
    print(path)
if __name__=='__main__':main()
