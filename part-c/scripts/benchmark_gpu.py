#!/usr/bin/env python3
"""CPU serial vs real CUDA, exact graph hashes/precision/K; no emulation or fallback."""
import argparse,csv,hashlib,json,math,os,pathlib,platform,random,shlex,subprocess,sys,time,uuid
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def parse(s):return dict(line.split(': ',1) for line in s.splitlines() if ': ' in line)
METRICS=['iterations','iteration_s','gpu_timeline_s','h2d_s','d2h_s','transfer_inclusive_s','load_s','gpu_setup_s','warmup_s','end_to_end_s','rank_sum','residual','validation_L1','gpu_name','compute_capability','cuda_runtime_version','cuda_driver_version','compute_s','allreduce_s','allgatherv_s','edge_imbalance']
HYBRID_KEYS=['ranks','threads','partition','schedule','chunk']

def load_graphs(manifest):
    graphs=json.loads(pathlib.Path(manifest).read_text());graphs=graphs.get('graphs',[]) if isinstance(graphs,dict) else graphs
    if not graphs:raise ValueError('no graphs selected')
    if len({g['graph_id'] for g in graphs})!=len(graphs):raise ValueError('duplicate graph IDs')
    for g in graphs:
        p=pathlib.Path(g['path']);p=p if p.is_absolute() else ROOT/p
        actual=sha(p)
        if g.get('sha256') and actual!=g['sha256']:raise ValueError(f'graph SHA256 mismatch: {g["graph_id"]}')
        g['resolved_path']=str(p.resolve());g['sha256']=actual
        with p.open() as f:
            if list(map(int,f.readline().split()))!=[g['N'],g['E']]:raise ValueError('graph header N/E mismatch')
    return graphs

def check_affinity(prefix,ranks,threads):
    seen=set();hosts=set()
    for rank in range(ranks):
        p=pathlib.Path(f'{prefix}.rank{rank}.json');m=json.loads(p.read_text());cpus=set(m['cpus'])
        if m['rank']!=rank or len(cpus)<threads or seen&cpus:raise ValueError('CPU masks overlap or cannot fit requested threads')
        seen|=cpus;hosts.add(m['hostname'])
    if hosts!={platform.node()}:raise ValueError('CPU/GPU comparison must run on the same single node')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest');ap.add_argument('--out',required=True);ap.add_argument('--site',choices=['Atlas','Vanda','local'],required=True)
    ap.add_argument('--hardware-tag',required=True);ap.add_argument('--build-tag',default='O3-cxx17-fp64');ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--K',type=int,default=50);ap.add_argument('--block-sizes',default='256');ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--warmup',type=int,default=2);ap.add_argument('--bin-dir',default=str(ROOT/'bin'))
    ap.add_argument('--hybrid-configs',help='Optional B-compatible hybrid configurations, rerun on this GPU node')
    ap.add_argument('--cpu-cores',type=int,help='Actual allocated CPU cores; required with --hybrid-configs');a=ap.parse_args()
    if a.repeats<1 or a.K<1 or a.warmup<0:raise ValueError('invalid repetition/iteration count')
    blocks=list(map(int,a.block_sizes.split(',')))
    if not blocks or len(set(blocks))!=len(blocks) or any(b not in [64,128,256,512] for b in blocks):raise ValueError('invalid or duplicate block size')
    bins=pathlib.Path(a.bin_dir).resolve()
    if not (bins/'pagerank_cuda').exists():raise FileNotFoundError('real CUDA executable required; no CPU fallback')
    hybrid=json.loads(pathlib.Path(a.hybrid_configs).read_text()) if a.hybrid_configs else []
    launcher=shlex.split(os.getenv('MPIEXEC','mpirun'));template=shlex.split(os.getenv('MPIEXEC_ARGS',''))
    if a.hybrid_configs and (not hybrid or not a.cpu_cores or not launcher or not template):
        raise ValueError('Hybrid requires nonempty configs, --cpu-cores, MPIEXEC and explicit MPIEXEC_ARGS with binding')
    for c in hybrid:
        if (c['ranks']<1 or c['threads']<1 or c['ranks']*c['threads']>a.cpu_cores or
            c['partition'] not in ['vertex','edge'] or c['schedule'] not in ['static','dynamic'] or c.get('chunk',0)<0):
            raise ValueError(f'invalid hybrid config: {c}')
        c.setdefault('chunk',64 if c['schedule']=='dynamic' else 0)
        if c['schedule']=='dynamic' and c['chunk']==0:c['chunk']=64
    if len({tuple(c[k] for k in HYBRID_KEYS) for c in hybrid})!=len(hybrid):raise ValueError('duplicate hybrid configs')
    required=['pagerank_reference','pagerank_cuda']+(['pagerank_hybrid'] if hybrid else [])
    for exe in required:
        if not os.access(bins/exe,os.X_OK):raise FileNotFoundError(f'Executable required: {bins/exe}')
    graphs=load_graphs(a.manifest)
    out=pathlib.Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.csv').exists():raise FileExistsError('use a fresh output directory')
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*')) if p.is_file()}
    sourcehash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    meta=dict(site=a.site,hardware_tag=a.hardware_tag,build_tag=a.build_tag,source_sha256=sourcehash,sources=sources,graphs=graphs,
      binary_sha256={x:sha(bins/x) for x in required},job_id=os.getenv('PBS_JOBID',os.getenv('SLURM_JOB_ID','none')),
      measurement_id=uuid.uuid4().hex,hostname=platform.node(),configuration=vars(a),hybrid_configs=hybrid,
      MPIEXEC=launcher,MPIEXEC_ARGS=template,OMP_environment={k:v for k,v in os.environ.items() if k.startswith('OMP_')},
      parent_cpu_mask=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
      CPU_PREFIX=os.getenv('CPU_PREFIX',''),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES','unset'))
    meta['workflow_sha256']={name:sha(ROOT/'scripts'/name) for name in
                             ['benchmark_gpu.py','analyze_gpu.py','cpu_affinity_exec.py']}
    for command,name in [(['nvidia-smi','-q'],'nvidia_smi.txt'),(['nvcc','--version'],'nvcc_version.txt'),(['lscpu'],'cpu_topology.txt')]:
        try:p=subprocess.run(command,capture_output=True,text=True);(out/name).write_text(p.stdout+p.stderr)
        except FileNotFoundError:(out/name).write_text('command unavailable; solver still must report actual CUDA device\n')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    variants=[('cpu_serial','none',0,dict(ranks=1,threads=1,partition='none',schedule='none',chunk=0)),
              *[('cuda',m,b,dict(ranks=0,threads=0,partition='none',schedule='none',chunk=0)) for m in ['thread','warp'] for b in blocks],
              *[('cpu_hybrid','none',0,c) for c in hybrid]]
    jobs=[(g,backend,mapping,b,r,c) for g in graphs for r in range(1,a.repeats+1) for backend,mapping,b,c in variants]
    random.Random(20261103).shuffle(jobs)
    columns=['run_id','measurement_id','hostname','site','hardware_tag','build_tag','source_sha256','graph_id','graph_sha256','model','seed','N','E','K','precision','backend','mapping','block_size',*HYBRID_KEYS,'cpu_cores','repeat','launcher_wall_s','job_id',*METRICS]
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();f.flush()
        for i,(g,backend,mapping,block,repeat,c) in enumerate(jobs):
            rid=f'{i+1:05d}_{g["graph_id"]}_{backend}_{mapping}_b{block}_r{repeat}'
            env=os.environ.copy()
            cmd=[str(bins/('pagerank_cuda' if backend=='cuda' else 'pagerank_reference')),g['resolved_path'],str(a.K)]
            if backend=='cuda':cmd+=['--mapping',mapping,'--block-size',str(block),'--device',str(a.device),'--warmup',str(a.warmup),'--validate']
            elif backend=='cpu_hybrid':
                env.update(OMP_NUM_THREADS=str(c['threads']),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',OMP_DISPLAY_AFFINITY='TRUE')
                cmd=[str(bins/'pagerank_hybrid'),g['resolved_path'],str(a.K),'--partition',c['partition'],'--schedule',c['schedule'],
                     '--profile',str(out/(rid+'.ranks.csv')),'--dump',str(out/(rid+'.rank.txt'))]
                if c['chunk']:cmd+=['--chunk',str(c['chunk'])]
                cmd=[*launcher,*[s.format(**c) for s in template],'-np',str(c['ranks']),sys.executable,
                     str(ROOT/'scripts/cpu_affinity_exec.py'),str(out/(rid+'.affinity')),*cmd]
            elif hybrid and not os.getenv('CPU_PREFIX'):
                cmd=[*launcher,*[s.format(ranks=1,threads=1) for s in template],'-np','1',sys.executable,
                     str(ROOT/'scripts/cpu_affinity_exec.py'),str(out/(rid+'.affinity')),*cmd]
            else:cmd=shlex.split(os.getenv('CPU_PREFIX',''))+cmd
            (out/(rid+'.command.json')).write_text(json.dumps(cmd)+'\n');t=time.perf_counter()
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=3600,env=env);wall=time.perf_counter()-t
            (out/(rid+'.stdout')).write_text(p.stdout);(out/(rid+'.stderr')).write_text(p.stderr)
            if p.returncode:raise RuntimeError(f'{rid}: failed, inspect stderr; no CPU fallback')
            m=parse(p.stdout)
            if backend=='cpu_hybrid':
                check_affinity(out/(rid+'.affinity'),c['ranks'],c['threads'])
                if int(m['ranks'])!=c['ranks'] or int(m['threads'])!=c['threads']:raise ValueError('actual CPU layout differs')
                # Independent same-graph FP64 full-vector comparison, outside reported solver timing.
                oracle=subprocess.run([str(bins/'pagerank_reference'),g['resolved_path'],str(a.K),'--dump',str(out/(rid+'.reference.txt'))],capture_output=True,text=True,check=True)
                def vector(path):
                    return [float(line.split()[-1]) for line in path.read_text().splitlines() if line.strip()]
                actual=vector(out/(rid+'.rank.txt'));reference=vector(out/(rid+'.reference.txt'))
                if len(actual)!=g['N'] or len(reference)!=g['N']:raise ValueError('invalid rank dump length')
                m['validation_L1']=sum(abs(x-y) for x,y in zip(actual,reference))
                if not math.isfinite(m['validation_L1']) or m['validation_L1']>=1e-8:raise ValueError('hybrid vector validation failed')
                m.update(backend=backend,precision='fp64')
                # The L1 diagnostic remains in CSV; avoid retaining hundreds of MB of duplicate vectors.
                (out/(rid+'.rank.txt')).unlink();(out/(rid+'.reference.txt')).unlink()
                (out/(rid+'.vector_validation.json')).write_text(json.dumps(dict(N=g['N'],K=a.K,graph_sha256=g['sha256'],L1=m['validation_L1']))+'\n')
            elif backend=='cpu_serial' and hybrid and not os.getenv('CPU_PREFIX'):
                check_affinity(out/(rid+'.affinity'),1,1)
            if m['backend']!=backend or m['precision']!='fp64' or int(m['iterations'])!=a.K:raise ValueError('backend/precision/K mismatch')
            if int(m['N'])!=g['N'] or int(m['E'])!=g['E']:raise ValueError('solver graph size mismatch')
            for key in ['iteration_s','rank_sum','residual']:
                if not math.isfinite(float(m[key])):raise ValueError('nonfinite result')
            if float(m['iteration_s'])<=0:raise ValueError('nonpositive runtime')
            if abs(float(m['rank_sum'])-1)>1e-8:raise ValueError('rank sum check failed')
            if backend=='cuda' and not 0<=float(m['validation_L1'])<1e-8:raise ValueError('GPU full-vector validation not passed')
            row={k:m.get(k,'') for k in columns};row.update(run_id=rid,site=a.site,hardware_tag=a.hardware_tag,build_tag=a.build_tag,
              source_sha256=sourcehash,graph_id=g['graph_id'],graph_sha256=g['sha256'],model=g.get('model','unknown'),seed=g.get('seed','unknown'),
              N=g['N'],E=g['E'],K=a.K,backend=backend,mapping=mapping,block_size=block,repeat=repeat,precision='fp64',launcher_wall_s=wall,
              measurement_id=meta['measurement_id'],hostname=meta['hostname'],job_id=meta['job_id'],cpu_cores=c['ranks']*c['threads'],**c)
            writer.writerow(row);f.flush();print(f'{i+1}/{len(jobs)} {rid} loop_s={m["iteration_s"]}',flush=True)
if __name__=='__main__':main()
