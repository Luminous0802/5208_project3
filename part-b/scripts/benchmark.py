#!/usr/bin/env python3
"""Fixed-K randomized repeated measurements, preserving stdout, stderr and per-rank data."""
import argparse, csv, datetime, hashlib, json, math, os, pathlib, platform, random, shlex, subprocess, time
from prepare_experiments import digest
ROOT=pathlib.Path(__file__).resolve().parents[1]

def fields(stdout):
    return dict(line.split(': ',1) for line in stdout.splitlines() if ': ' in line)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('--configs',default=str(ROOT/'scripts/configs_12cores.json'))
    ap.add_argument('--out',required=True);ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--K',type=int,default=50)
    ap.add_argument('--bin-dir',default=str(ROOT/'bin'));ap.add_argument('--site',choices=['local','Atlas','Vanda'],default='local');ap.add_argument('--max-cores',type=int,required=True)
    ap.add_argument('--experiments',default='E1,E2,E3,E4');ap.add_argument('--order-seed',type=int,default=20261103);a=ap.parse_args()
    if a.K<=0 or a.repeats<=0:raise ValueError('positive K and repeats required')
    bin_dir=pathlib.Path(a.bin_dir).resolve()
    out=pathlib.Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    csvpath=out/'runs.csv'
    if csvpath.exists():raise FileExistsError(f'Choose a new output directory: {csvpath}')
    graphs=json.loads(pathlib.Path(a.manifest).read_text());wanted=set(a.experiments.split(','))
    graphs=[g for g in graphs if set(g['experiments'])&wanted]
    configs=json.loads(pathlib.Path(a.configs).read_text())
    if not graphs or not configs:raise ValueError('empty graph/config selection')
    for c in configs:
        if c['ranks']<1 or c['threads']<1 or c['ranks']*c['threads']>a.max_cores:raise ValueError(f'invalid core request: {c}')
        if c['version'] not in ['V0','V1','V2','V3']:raise ValueError('invalid version')
        if c['version']=='V0' and (c['ranks']!=1 or c['threads']!=1):raise ValueError('V0 requires 1x1')
        if c['version']=='V1' and c['threads']!=1:raise ValueError('V1 requires T=1')
        if c['version'] in ['V2','V3'] and c.get('schedule') not in ['static','dynamic']:raise ValueError('missing hybrid schedule')
    for g in graphs:
        actual=digest(g['path'])
        if g.get('sha256') and actual!=g['sha256']:raise ValueError('graph SHA256 mismatch')
        g['sha256']=actual
    sourcehash=hashlib.sha256()
    sources=sorted(list((ROOT/'src').glob('*'))+list((ROOT/'baseline_a').glob('*.cpp')))
    for path in sources:sourcehash.update(path.name.encode()+b'\0'+path.read_bytes())
    rev=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    launcher=shlex.split(os.environ.get('MPIEXEC','mpirun'))
    template=shlex.split(os.environ.get('MPIEXEC_ARGS',''))
    metadata=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),site=a.site,
        platform=platform.platform(),hostname=platform.node(),job_id=os.getenv('PBS_JOBID',os.getenv('SLURM_JOB_ID','none')),
        source_sha256=sourcehash.hexdigest(),commit_hash=rev.stdout.strip() if rev.returncode==0 else 'unversioned',
        binary_sha256={p.name:digest(p) for p in bin_dir.iterdir() if p.is_file()},
        configuration=vars(a),MPIEXEC=launcher,MPIEXEC_ARGS=template,
        environment={k:v for k,v in os.environ.items() if k.startswith(('OMP_','PBS_','SLURM_'))},graphs=graphs,configs=configs)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    jobs=[(g,c,repeat) for repeat in range(1,a.repeats+1) for g in graphs for c in configs]
    random.Random(a.order_seed).shuffle(jobs)
    columns=['run_id','site','graph_id','graph_sha256','seed','N','E','model','experiments','version','partition','schedule','chunk',
        'ranks','threads','cores','K','iterations','repeat','iteration_s','compute_s','allreduce_s','allgatherv_s','load_s','partition_s',
        'end_to_end_s','launcher_wall_s','edge_imbalance','rank_sum','residual','job_id','hostname','source_sha256','commit_hash']
    with csvpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();f.flush()
        for i,(g,c,rep) in enumerate(jobs):
            run_id=f'{i+1:05d}_{g["graph_id"]}_{c["version"]}_p{c["ranks"]}_t{c["threads"]}_{c.get("schedule","none")}_c{c.get("chunk",0)}_r{rep}'
            env=os.environ.copy();env.update(OMP_NUM_THREADS=str(c['threads']),OMP_DYNAMIC='FALSE')
            env.setdefault('OMP_PROC_BIND','close');env.setdefault('OMP_PLACES','cores')
            hybrid=c['version'] in ['V2','V3'];part='edge' if c['version']=='V3' else 'vertex'
            exe='pagerank_hybrid' if hybrid else ('pagerank' if c['version']=='V0' else 'pagerank_mpi')
            command=[str(bin_dir/exe),g['path'],str(a.K)]
            if hybrid:command+=['--partition',part,'--schedule',c['schedule'],'--profile',str(out/(run_id+'.ranks.csv'))]
            if hybrid and c.get('chunk',0):command+=['--chunk',str(c['chunk'])]
            # Launch serial through the same launcher for controlled CPU binding too.
            if launcher:
                extra=[s.format(ranks=c['ranks'],threads=c['threads']) for s in template]
                command=[*launcher,*extra,'-np',str(c['ranks']),*command]
            (out/(run_id+'.command.json')).write_text(json.dumps(command)+'\n')
            start=time.perf_counter();p=subprocess.run(command,text=True,capture_output=True,env=env);wall=time.perf_counter()-start
            (out/(run_id+'.stdout')).write_text(p.stdout);(out/(run_id+'.stderr')).write_text(p.stderr)
            if p.returncode:raise RuntimeError(f'{run_id} failed: see stderr')
            m=fields(p.stdout)
            for key in ['iterations','iteration_s','rank_sum','residual']:
                if key not in m:raise ValueError(f'missing metric {key}')
            if not all(math.isfinite(float(m[k])) for k in ['iteration_s','rank_sum','residual']):raise ValueError('non-finite metrics')
            if float(m['iteration_s'])<0:raise ValueError('negative runtime')
            if hybrid and (int(m['N'])!=g['N'] or int(m['E'])!=g['E']):raise ValueError('graph/manifest N/E mismatch')
            if int(m['iterations'])!=a.K or abs(float(m['rank_sum'])-1)>1e-8:raise ValueError('K/rank_sum validation failed')
            if hybrid and int(m['threads'])!=c['threads']:raise ValueError('actual thread count differs')
            row={k:m.get(k,'') for k in columns}
            row.update(run_id=run_id,site=a.site,graph_id=g['graph_id'],graph_sha256=g['sha256'],seed=g['seed'],N=g['N'],E=g['E'],
                model=g['model'],experiments=';'.join(g['experiments']),version=c['version'],partition='serial' if c['version']=='V0' else part,
                schedule=c.get('schedule','none'),chunk=m.get('chunk',0),ranks=c['ranks'],threads=c['threads'],cores=c['ranks']*c['threads'],K=a.K,repeat=rep,
                launcher_wall_s=wall,job_id=metadata['job_id'],hostname=platform.node(),source_sha256=metadata['source_sha256'],commit_hash=metadata['commit_hash'])
            writer.writerow(row);f.flush();print(f'{i+1}/{len(jobs)} {run_id} iteration_s={m["iteration_s"]}',flush=True)
    # Graph statistics are computed outside timed solver runs, for each used partition/rank pair.
    seen=set()
    for g,c,_ in jobs:
        if c['version']=='V0':continue
        part='edge' if c['version']=='V3' else 'vertex';key=(g['graph_id'],c['ranks'],part)
        if key in seen:continue
        seen.add(key)
        proc=subprocess.run([str(bin_dir/'graph_stats'),g['path'],str(c['ranks']),part],text=True,capture_output=True,check=True)
        data=json.loads(proc.stdout);data.update(graph_id=g['graph_id'],sha256=g['sha256'])
        (out/f'{g["graph_id"]}_p{c["ranks"]}_{part}.stats.json').write_text(json.dumps(data,indent=2)+'\n')
    print(csvpath)
if __name__=='__main__':main()
