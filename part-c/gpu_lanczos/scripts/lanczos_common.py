"""CSV helpers only; importing this module never launches a solver."""
import csv
import math
from pathlib import Path
import statistics

IDENTITY = ['site','hostname','job_id','measurement_id','hardware_tag','source_sha256','binary_sha256','precision','stage']
GROUP = IDENTITY + ['model','graph_sha256','N','input_E','seed','operator','reorth','budget','residual_tolerance','variant','ranks','threads']
TIMES = ['solve_s','iteration_s','end_to_end_s','load_s','gpu_setup_s','warmup_s','setup_h2d_s',
         'iteration_h2d_s','iteration_d2h_s','solve_h2d_s','solve_d2h_s','iteration_gpu_event_s','solve_gpu_event_s',
         'iteration_allreduce_s','iteration_allgatherv_s']
VARIANTS = [dict(name='cpu_serial',binary='lanczos_serial',ranks=1,threads=1,gpu=False),
            dict(name='cpu_mpi2_omp2',binary='lanczos_cpu_mpi',ranks=2,threads=2,gpu=False),
            dict(name='gpu_mpi1',binary='lanczos_cuda_mpi',ranks=1,threads=2,gpu=True),
            dict(name='gpu_mpi2_shared',binary='lanczos_cuda_mpi',ranks=2,threads=2,gpu=True)]
THRESHOLDS = dict(norm_error=1e-10,dense_residual_agreement=1e-10,rayleigh_agreement=1e-10,eigenvalue_error=1e-8,
                  converged_residual=1e-8,orthogonality=1e-9,one_step_L1=1e-9,one_step_Linf=1e-10)

def write_csv(path, rows):
    if not rows:raise ValueError('No rows to write')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(rows):
    groups={}
    for row in rows:groups.setdefault(tuple(str(row[k]) for k in GROUP),[]).append(row)
    result=[]
    for key,items in sorted(groups.items()):
        out=dict(zip(GROUP,key));out['samples']=len(items)
        out['actual_steps_set']=';'.join(map(str,sorted({int(r['steps']) for r in items})))
        out['spmv_count_set']=';'.join(map(str,sorted({int(r['spmv_count']) for r in items})))
        out['all_binding_checks_passed']=all(str(r['binding_verified'])=='True' for r in items)
        for field in TIMES:
            values=[float(r[field]) for r in items if str(r.get(field,''))!='']
            for suffix,fn in [('median',statistics.median),('min',min),('max',max)]:out[field+'_'+suffix]=fn(values) if values else ''
        out['lambda_median']=statistics.median(float(r['lambda_ritz']) for r in items)
        out['residual_median']=statistics.median(float(r['direct_residual']) for r in items)
        out['residual_max']=max(float(r['direct_residual']) for r in items)
        out['all_reached_tol']=all(float(r['direct_residual'])<float(r['residual_tolerance']) for r in items)
        out['gpu_name']=items[0].get('gpu_name','');out['gpu_uuid']=items[0].get('gpu_uuid','')
        result.append(out)
    fields=IDENTITY+['model','graph_sha256','N','input_E','seed','operator','reorth','budget','residual_tolerance']
    def match(r):return tuple(str(r[k]) for k in fields)
    serial={match(r):r for r in result if r['variant']=='cpu_serial'}
    hybrid={match(r):r for r in result if r['variant']=='cpu_mpi2_omp2'}
    for out in result:
        notes=[]
        for label,base in [('serial',serial.get(match(out))),('cpu_mpi2_omp2',hybrid.get(match(out)))]:
            compatible=base is not None and base['actual_steps_set']==out['actual_steps_set'] and ';' not in out['actual_steps_set'] and base['spmv_count_set']==out['spmv_count_set']
            for metric in ['solve_s','end_to_end_s']:
                out[metric+'_speedup_vs_'+label]=(base[metric+'_median']/out[metric+'_median']) if compatible else ''
            if not compatible:notes.append(label+': no compatible same-step baseline')
        out['comparison_note']='; '.join(notes) or 'Same-job same-graph FP64 lazy full-reorth fixed-budget comparison; solve and post-MPI-init application time are separate'
    return result

def validate_rows(rows):
    if not rows:raise ValueError('Empty runs.csv')
    seen=set();repetitions=set();known={v['name']:v for v in VARIANTS};devices=set()
    for r in rows:
        if r['site']!='Vanda' or not r['job_id'] or not r['job_id'][0].isdigit():raise ValueError('Require Vanda PBS measurements')
        if r['precision']!='fp64' or r['operator']!='lazy' or r['reorth']!='full':raise ValueError('Incompatible numerical protocol')
        if str(r['binding_verified'])!='True':raise ValueError('Missing CPU-mask checks')
        identity=(r['measurement_id'],r['run_id'])
        if identity in seen:raise ValueError('Duplicate run')
        seen.add(identity)
        if r['variant'] not in known:raise ValueError('Unknown variant')
        expected=known[r['variant']]
        if int(r['ranks'])!=expected['ranks'] or int(r['threads'])!=expected['threads']:raise ValueError('Variant layout mismatch')
        repetition=(r['model'],r['graph_sha256'],r['budget'],r['variant'],r['repeat'])
        if repetition in repetitions:raise ValueError('Repeated configuration/repeat identity')
        repetitions.add(repetition)
        if int(r['repeat'])<1 or not 1<=int(r['steps'])<=min(int(r['budget']),int(r['N'])) or int(r['spmv_count'])!=int(r['steps'])+1:raise ValueError('Invalid repeat/step/SpMV count')
        for key in ['solve_s','iteration_s','end_to_end_s']:
            if not math.isfinite(float(r[key])) or float(r[key])<=0:raise ValueError('Invalid '+key)
        if not math.isfinite(float(r['direct_residual'])) or float(r['direct_residual'])<0:raise ValueError('Invalid residual')
        if expected['gpu']:
            if int(r['gpu_kernel_calls'])<1 or not r['gpu_uuid'] or int(r['shared_physical_gpus'])!=1:raise ValueError('No real shared single-GPU execution evidence')
            devices.add((r['gpu_uuid'],r['gpu_name']))
    keys=IDENTITY+['N','input_E','seed','operator','reorth','residual_tolerance']
    if len({tuple(r[k] for k in keys) for r in rows})!=1:raise ValueError('Analyze one compatible job at a time')
    if len(devices)>1:raise ValueError('Different physical GPUs in one measurement')
