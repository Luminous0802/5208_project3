#!/usr/bin/env python3
"""Vanda-only MPI+CUDA Lanczos validation and measurement. No CPU fallback."""
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
import sys
import uuid

PROJECT=Path(__file__).resolve().parents[2]
MODULE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
from process_logging import run_logged
from lanczos_common import VARIANTS,THRESHOLDS,TIMES,summarize,write_csv,validate_rows

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parse(stdout):return dict(line.split(': ',1) for line in stdout.splitlines() if ': ' in line)
def vector(path,n):
    rows=[line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=n or [int(row[0]) for row in rows]!=list(range(n)):raise ValueError('Invalid full vector dump')
    answer=[float(row[1]) for row in rows]
    if not all(math.isfinite(x) for x in answer):raise ValueError('Nonfinite vector')
    return answer

def dense_lazy(n,edges):
    # Independent small dense operator: no C++ CSR/weight reuse.
    a=[[0.0]*n for _ in range(n)]
    for u,v in edges:a[u][v]=a[v][u]=1.0
    degree=[math.fsum(row) for row in a]
    return [[0.5*((a[i][j]/math.sqrt(degree[i]*degree[j]) if a[i][j] else 0.0)+(1.0 if i==j else 0.0)) for j in range(n)] for i in range(n)]

class Experiment:
    def __init__(self,a):
        self.a=a;self.out=a.out.expanduser().resolve();self.bins=a.bin_dir.expanduser().resolve()
        if not os.getenv('PBS_JOBID') or not os.getenv('PBS_NODEFILE'):raise RuntimeError('Run only inside the requested Vanda PBS allocation')
        if self.out.exists() and any(self.out.iterdir()):raise FileExistsError('Use a fresh output directory')
        self.out.mkdir(parents=True,exist_ok=True)
        if len(set(Path(os.environ['PBS_NODEFILE']).read_text().split()))!=1:raise RuntimeError('Exactly one PBS node is required')
        if not hasattr(os,'sched_getaffinity'):raise RuntimeError('Linux CPU affinity support required')
        self.allowed=set(os.sched_getaffinity(0));self.gpu_uuid=None
        mpi=Path(os.environ.get('MPIEXEC',''))
        if not mpi.is_file() or not os.access(mpi,os.X_OK):raise RuntimeError('An absolute MPICH Hydra path is required')
        self.launcher=[str(mpi),'-rmk','user','-launcher','fork','-hosts','localhost']
        for name in ['make_graph','lanczos_serial','lanczos_cpu_mpi','lanczos_cuda_mpi']:
            if not os.access(self.bins/name,os.X_OK):raise FileNotFoundError(self.bins/name)
        paths=[PROJECT/'src'/name for name in ['graph.hpp','symmetric.hpp','lanczos.cpp','make_graph.cpp']]
        paths += [p for folder in ['src','scripts','configs'] for p in sorted((MODULE/folder).rglob('*')) if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts]
        paths += [PROJECT/'scripts/process_logging.py',PROJECT/'scripts/cpu_affinity_exec.py',PROJECT/'scripts/run_vanda_gpu_lanczos.pbs',MODULE/'Makefile']
        self.sources={str(p.relative_to(PROJECT)):sha(p) for p in paths}
        binaries={name:sha(self.bins/name) for name in ['make_graph','lanczos_serial','lanczos_cpu_mpi','lanczos_cuda_mpi']}
        self.identity=dict(site='Vanda',hostname=platform.node(),job_id=os.environ['PBS_JOBID'],measurement_id=uuid.uuid4().hex,
                           hardware_tag=platform.node()+'_single_allocated_GPU',source_sha256=hashlib.sha256(json.dumps(self.sources,sort_keys=True).encode()).hexdigest(),
                           binary_sha256=hashlib.sha256(json.dumps(binaries,sort_keys=True).encode()).hexdigest(),precision='fp64',stage=a.stage)
        self.meta=dict(self.identity,status='STARTING',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files=self.sources,binaries=binaries,
                       configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},variants=VARIANTS,
                       environment={k:v for k,v in os.environ.items() if k.startswith(('OMP_','FI_','MPIR_','HYDRA_','CUDA_'))},
                       parent_cpu_mask=sorted(self.allowed),numerical_protocol=dict(operator='B=(I+S)/2',reorth='two-pass full',seed=20261029,budgets=[5,10,20,30,50],
                       tolerance=1e-8,tolerance_usage='Post-hoc accuracy target, not an early-stop rule'),
                       timing_contract=dict(solve_s='Recurrence + projected host eigensolve + GPU/CPU reconstruction + independent-form direct residual, including recurring host staging and MPI; excludes graph loading and pre-solve GPU allocation/warmup',
                       end_to_end_s='Application-internal post-MPI-init to completed numerical solve; GPU includes setup/warmup; excludes process startup/finalization, event readout and vector-file I/O',
                       event_s='Sum of per-kernel CUDA event intervals, per-rank maximum; includes possible scheduling/interference, not exclusive GPU compute or full application time',
                       transfer_s='Blocking host-observed staging intervals, may include queued work; per-rank maxima must not be added as percentages',
                       GPU_layout='One physical GPU shared by one or two MPI ranks; not multi-GPU scaling'),graphs=[],completed_runs=0)
        self.save()

    def save(self):
        (self.out/'provenance.json').write_text(json.dumps(self.meta,indent=2)+'\n')

    def invoke(self,variant,graph,budget,stem,dump=None,orth=False,timeout=180):
        stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
        prefix=Path(str(stem)+'.affinity')
        env=os.environ.copy();env.update(OMP_NUM_THREADS=str(variant['threads']),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores')
        solver=[str(self.bins/variant['binary']),str(graph),str(budget),'--operator','lazy','--seed','20261029','--reorth','full']
        if variant['gpu']:solver+=['--block-size','256']
        if orth:solver+=['--orthogonality','1']
        if dump is not None:solver+=['--dump',str(dump)]
        cmd=[*self.launcher,'-bind-to',f"core:{variant['threads']}",'-n',str(variant['ranks']),sys.executable,
             str(PROJECT/'scripts/cpu_affinity_exec.py'),str(prefix),*solver]
        result=run_logged(cmd,stem,env=env,timeout=timeout)
        used=set()
        for rank in range(variant['ranks']):
            m=json.loads(Path(f'{prefix}.rank{rank}.json').read_text());cpus=set(m['cpus'])
            if m['rank']!=rank or m['hostname']!=platform.node() or len(cpus)<variant['threads'] or used&cpus or not cpus<=self.allowed:raise RuntimeError('Invalid/overlapping MPI CPU masks')
            used|=cpus
        m=parse(result.stdout)
        if int(m['ranks'])!=variant['ranks'] or int(m['threads'])!=variant['threads'] or m['operator']!='lazy' or m['reorth']!='full':raise RuntimeError('Actual numerical layout differs')
        for key in ['solve_s','iteration_s','end_to_end_s','ritz_residual','lambda_max_ritz']:
            if not math.isfinite(float(m[key])):raise RuntimeError('Nonfinite '+key)
        if variant['gpu']:
            if m.get('backend')!='cuda_lanczos' or int(m.get('gpu_kernel_calls','0'))<1 or int(m.get('shared_physical_gpus','0'))!=1:raise RuntimeError('Real CUDA execution was not confirmed')
            if self.gpu_uuid is None:
                self.gpu_uuid=m['gpu_uuid']
                self.identity['hardware_tag']=platform.node()+'_'+m['gpu_name']
                self.meta['hardware_tag']=self.identity['hardware_tag']
                self.meta['gpu_device']={key:m[key] for key in ['gpu_name','gpu_uuid','compute_capability','cuda_runtime_version','cuda_driver_version','shared_physical_gpus']}
                self.save()
            if self.gpu_uuid!=m['gpu_uuid']:raise RuntimeError('GPU changed within the experiment')
        return m

    def validate(self):
        out=self.out/'validation';out.mkdir();inputs=out/'inputs';inputs.mkdir()
        (out/'thresholds.json').write_text(json.dumps(THRESHOLDS,indent=2)+'\n')
        cases={'zero':(3,[]),'single':(1,[]),'tiny':(2,[(0,1)]),'path_uneven':(11,[(i,i+1) for i in range(10)]),
               'cycle':(10,[(i,(i+1)%10) for i in range(10)]),'self_disconnected':(5,[(0,0),(1,2),(2,3)])}
        rows=[]
        try:
            for name,(n,edges) in cases.items():
                graph=inputs/(name+'.txt');graph.write_text(f'{n} {len(edges)}\n'+''.join(f'{u} {v}\n' for u,v in edges));b=dense_lazy(n,edges)
                reference={}
                for budget_tag,budget in [('one_step',1),('full',n)]:
                    for variant in VARIANTS:
                        label=f"{name}_{budget_tag}_{variant['name']}";dump=inputs/label
                        print('VALIDATION_START',label,flush=True)
                        m=self.invoke(variant,graph,budget,out/'raw'/label,dump=dump,orth=True,timeout=self.a.validation_timeout)
                        v=vector(Path(str(dump)+'.vector.txt'),n);lam=float(m['lambda_max_ritz'])
                        bv=[math.fsum(b[i][j]*v[j] for j in range(n)) for i in range(n)]
                        residual=math.sqrt(math.fsum((bv[i]-lam*v[i])**2 for i in range(n)))
                        rayleigh=math.fsum(v[i]*bv[i] for i in range(n));rayleigh_error=abs(rayleigh-lam)
                        norm_error=abs(math.fsum(x*x for x in v)-1);residual_error=abs(residual-float(m['ritz_residual']))
                        if norm_error>THRESHOLDS['norm_error'] or residual_error>THRESHOLDS['dense_residual_agreement'] or rayleigh_error>THRESHOLDS['rayleigh_agreement'] or float(m['orthogonality_max'])>THRESHOLDS['orthogonality']:raise RuntimeError(f"Independent check failed: {label}; norm={norm_error}, residual_agreement={residual_error}, Rayleigh_agreement={rayleigh_error}, orthogonality={m['orthogonality_max']}")
                        if budget_tag=='full' and (abs(lam-(1.0 if edges else 0.5))>THRESHOLDS['eigenvalue_error'] or residual>THRESHOLDS['converged_residual']):raise RuntimeError('Analytic eigenvalue/full residual failed: '+label)
                        if variant['name']=='cpu_serial':reference[budget_tag]=(lam,residual,v)
                        ref_lam,ref_res,ref_v=reference[budget_tag]
                        l1=linf=''
                        if budget_tag=='one_step':
                            sign=1 if math.fsum(x*y for x,y in zip(v,ref_v))>=0 else -1
                            error=[abs(x-sign*y) for x,y in zip(v,ref_v)];l1=math.fsum(error);linf=max(error)
                            if l1>THRESHOLDS['one_step_L1'] or linf>THRESHOLDS['one_step_Linf']:raise RuntimeError('One-step full-vector comparison failed: '+label)
                        if abs(lam-ref_lam)>THRESHOLDS['eigenvalue_error']:raise RuntimeError('Eigenvalue differs from serial: '+label)
                        rows.append(dict(site='Vanda',job_id=self.identity['job_id'],validation_id=self.identity['measurement_id']+'_validation',graph=name,graph_sha256=sha(graph),N=n,input_E=len(edges),
                            variant=variant['name'],ranks=variant['ranks'],threads=variant['threads'],budget_tag=budget_tag,budget=budget,steps=int(m['steps']),lambda_ritz=lam,dense_residual=residual,
                            residual_report_error=residual_error,rayleigh_quotient=rayleigh,rayleigh_error=rayleigh_error,norm_error=norm_error,orthogonality=float(m['orthogonality_max']),one_step_L1=l1,one_step_Linf=linf,
                            gpu_name=m.get('gpu_name',''),gpu_uuid=m.get('gpu_uuid',''),status='PASS'))
                        write_csv(out/'comparisons.csv',rows)
                        print('VALIDATION_PASS',label,'dense_residual=',residual,flush=True)
            if len(rows)!=48:raise RuntimeError('Incomplete validation')
            summary=dict(status='PASS',site='Vanda',job_id=self.identity['job_id'],comparisons=len(rows),thresholds=THRESHOLDS,full_vectors_retained=True,
                         max_dense_residual_agreement=max(r['residual_report_error'] for r in rows),max_rayleigh_error=max(r['rayleigh_error'] for r in rows),max_norm_error=max(r['norm_error'] for r in rows),
                         note='Full vectors are compared at one step only; eigenvectors of repeated eigenvalues are not unique. Converged cases use independent dense residuals and analytic eigenvalues.')
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        except Exception as error:
            (out/'summary.json').write_text(json.dumps(dict(status='FAILED',completed_comparisons=len(rows),error=str(error)),indent=2)+'\n');raise
        print('GPU_LANCZOS_VALIDATION_PASS: 48 comparisons',flush=True)

    def prepare(self):
        catalog=json.loads((MODULE/'configs/graphs.json').read_text());generator=sha(PROJECT/'src/make_graph.cpp')
        inputs=self.out/'inputs';inputs.mkdir()
        graphs=[]
        selected=catalog['graphs'] if self.a.stage=='full' else catalog['graphs'][:1]
        for spec in selected:
            if generator!=catalog['generator_sha256']:raise RuntimeError('Graph generator source changed; do not silently change the shared input')
            graph=inputs/(spec['model']+'.txt')
            cmd=[str(self.bins/'make_graph'),spec['model'],str(spec['N']),str(spec['input_E']),str(spec['seed']),str(graph)]
            run_logged(cmd,self.out/'raw'/('generate_'+spec['model']),timeout=180)
            if sha(graph)!=spec['sha256']:raise RuntimeError('Generated graph differs from the prior Vanda eigen input')
            graphs.append(dict(spec,path=str(graph)))
        self.meta['graphs']=graphs;self.save();return graphs

    def run(self):
        try:
            self.meta['status']='VALIDATING';self.save();self.validate()
            graphs=self.prepare();budgets=[5,10,20,30,50] if self.a.stage=='full' else [5,10]
            repeats=5 if self.a.stage=='full' else 1
            jobs=[(g,budget,variant,repeat) for g in graphs for budget in budgets for variant in VARIANTS for repeat in range(1,repeats+1)]
            random.Random(20261110).shuffle(jobs);self.meta.update(status='MEASURING',planned_runs=len(jobs),repeats=repeats);self.save()
            rows=[]
            for i,(g,budget,variant,repeat) in enumerate(jobs,1):
                label=f"{i:04d}_{g['model']}_b{budget}_{variant['name']}_r{repeat}"
                m=self.invoke(variant,g['path'],budget,self.out/'raw'/label,timeout=self.a.timeout)
                if int(m['N'])!=g['N'] or int(m['input_E'])!=g['input_E'] or not 1<=int(m['steps'])<=budget:raise RuntimeError('Graph/step count mismatch')
                row=dict(self.identity,run_id=label,model=g['model'],graph_sha256=g['sha256'],N=g['N'],input_E=g['input_E'],seed=g['seed'],operator='lazy',reorth='full',budget=budget,residual_tolerance=1e-8,
                         variant=variant['name'],ranks=variant['ranks'],threads=variant['threads'],binding_verified=True,repeat=repeat,steps=int(m['steps']),spmv_count=int(m['steps'])+1,undirected_edges=int(m['undirected_edges']),
                         lambda_ritz=float(m['lambda_max_ritz']),direct_residual=float(m['ritz_residual']),
                         **{key:m.get(key,'') for key in TIMES},
                         gpu_kernel_calls=m.get('gpu_kernel_calls',''),gpu_name=m.get('gpu_name',''),gpu_uuid=m.get('gpu_uuid',''),shared_physical_gpus=m.get('shared_physical_gpus',''),
                         cuda_runtime_version=m.get('cuda_runtime_version',''),cuda_driver_version=m.get('cuda_driver_version',''),compute_capability=m.get('compute_capability',''))
                if not variant['gpu']:
                    row['iteration_allreduce_s']=m.get('allreduce_s','');row['iteration_allgatherv_s']=m.get('allgatherv_s','')
                rows.append(row);write_csv(self.out/'runs.csv',rows)
                self.meta.update(completed_runs=len(rows));self.save()
                print(f"{i}/{len(jobs)} {label} solve_s={m['solve_s']} residual={m['ritz_residual']}",flush=True)
            validate_rows(rows);summary=summarize(rows);write_csv(self.out/'summary.csv',summary)
            expected=300 if self.a.stage=='full' else 8
            expected_configurations=60 if self.a.stage=='full' else 8
            if len(rows)!=expected or len(summary)!=expected_configurations or any(r['samples']!=repeats for r in summary):raise RuntimeError('Incomplete measurement plan')
            self.meta.update(status='COMPLETE',gpu_uuid=self.gpu_uuid,configurations=len(summary),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());self.save()
            print(f"GPU_LANCZOS_COMPLETE: stage={self.a.stage} runs={len(rows)} configurations={len(summary)} results={self.out}",flush=True)
        except Exception as error:
            self.meta.update(status='FAILED',error=str(error));self.save();raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bin-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=['pilot','full'],default='full');ap.add_argument('--timeout',type=int,default=300);ap.add_argument('--validation-timeout',type=int,default=90)
    a=ap.parse_args()
    if a.timeout<=0 or a.validation_timeout<=0:ap.error('Positive timeouts required')
    Experiment(a).run()
if __name__=='__main__':main()
