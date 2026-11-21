"""Run shared-score synthetic experiments. CPU-only, no model downloads."""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poppim.encoding.bitpacking import encode_query, encode_keys
from poppim.scoring.reference import score_encoded, dense_encoded_oracle
from poppim.memory.bank_mapper import map_banks
from poppim.simulator.functional import run_retrieval
from poppim.simulator.cost_model import HardwareConfig, estimate
from poppim.simulator.trace_generator import write_trace
from poppim.topk.host_topk import accuracy

MODES = ('host','bank_local','bank_adaptive','near_memory')


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def environment():
    def read(path):
        try: return Path(path).read_text().strip()
        except OSError: return None
    return {'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'os_release':read('/etc/os-release'),
            'logical_cpus_visible':os.cpu_count(), 'cpu_affinity_count':len(os.sched_getaffinity(0)),
            'cgroup_memory_max':read('/sys/fs/cgroup/memory.max'),
            'cgroup_cpu_max':read('/sys/fs/cgroup/cpu.max'),
            'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
            'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')}


def synthetic(context, banks, seed, scenario='iid', mapping='striped', q_heads=4, kv_heads=2):
    if q_heads % kv_heads:
        raise ValueError('invalid GQA counts')
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(1,q_heads,128)).astype(np.float16)
    k = rng.normal(size=(kv_heads,context,128)).astype(np.float16)
    if scenario == 'bank_cluster':
        hot = map_banks(context,banks,mapping) < max(1,banks//8)
        for kv in range(kv_heads):
            # Synthetic importance concentration for first query of each GQA group.
            k[kv,hot] = (k[kv,hot].astype(np.float32) +
                         3*q[0,kv*(q_heads//kv_heads)].astype(np.float32)).astype(np.float16)
    elif scenario != 'iid':
        raise ValueError('scenario must be iid or bank_cluster')
    return q,k


def run_case(context=4096,budget=512,banks=32,seed=42,scenario='iid',
             mapping='striped',modes=MODES,repeats=3,chunk_size=1024,hw=None):
    if not 1 <= budget <= context or repeats < 1:
        raise ValueError('need 1 <= budget <= context and positive repeats')
    hw = hw or HardwareConfig()
    q,k = synthetic(context,banks,seed,scenario,mapping)
    start = time.perf_counter_ns()
    qe,ke = encode_query(q,64),encode_keys(k,64,64)
    encode_ns = time.perf_counter_ns()-start
    start = time.perf_counter_ns()
    scores = score_encoded(qe,ke,chunk_size=chunk_size)[0]
    score_cpu_ns = time.perf_counter_ns()-start
    dense = dense_encoded_oracle(qe,ke,chunk_size=chunk_size)[0]
    digest = hashlib.sha256(scores.tobytes()).hexdigest()
    common = dict(context=context,budget=budget,banks=banks,seed=seed,scenario=scenario,
                  mapping=mapping,score_sha256=digest)
    refs = [run_retrieval(s,budget,'host',banks,mapping)['indices'] for s in scores]
    result = {name:[] for name in ['score_correctness','topk_accuracy','bank_distribution','cpu_functional_benchmark',
                                 'latency_comparison','traffic_comparison','hardware_cost']}
    selected_by_mode = {}
    costs = []
    for h,(s,d) in enumerate(zip(scores,dense)):
        error = np.abs(s.astype(float)-d)
        tol = .501*np.spacing(np.abs(s)).astype(float)+5e-6*(1+np.abs(d))
        row = dict(common,head=h,source='Actual CPU Measurement; FP64 same-metadata oracle',
                   max_abs_error=float(error.max()),rmse=float(np.sqrt(np.mean(error**2))),
                   mean_abs_error=float(error.mean()),tolerance_failures=int(np.sum(error > tol)),
                   score_dtype=str(s.dtype),q_coeff_dtype=str(qe.low.dtype),k_coeff_dtype=str(ke.low.dtype),
                   fp16_oracle_storage_agreement=float(np.mean(s == d.astype(np.float16))),
                   cuda_comparison='not performed',encoding_cpu_ns=encode_ns,scoring_cpu_ns=score_cpu_ns)
        if row['tolerance_failures']:
            raise AssertionError(f'score oracle failed: {row}')
        result['score_correctness'].append(row)
    for mode in modes:
        # Warm-up outside samples. CPU medians are NOT used in hardware model.
        def select():
            return [run_retrieval(s,budget,mode,banks,mapping) for s in scores]
        select()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            outputs = select()
            samples.append(time.perf_counter_ns()-start)
        selected_by_mode[mode] = outputs
        for h,out in enumerate(outputs):
            metric = accuracy(refs[h],out['indices'])
            if mode in ('host','near_memory'):
                np.testing.assert_array_equal(refs[h],out['indices'])
            result['topk_accuracy'].append(dict(common,mode=mode,head=h,**metric,
                source='Actual CPU Measurement; synthetic encoded Q/K',
                selected_count=len(out['indices']),
                control_bytes=out.get('control_bytes',0)))
        quota = np.stack([o['quotas'] for o in outputs]) if mode.startswith('bank_') else None
        control_bytes = sum(o.get('control_bytes',0) for o in outputs)
        rounds = max(o.get('synchronization_rounds',0) for o in outputs)
        cost = estimate(context,budget,banks,mapping=mapping,mode=mode,
                        quotas=quota,control_bytes=control_bytes,synchronization_rounds=rounds,hw=hw)
        costs.append(cost)
        result['latency_comparison'].append(dict(common,**cost,
            cpu_topk_median_ns=float(np.median(samples)),cpu_topk_min_ns=min(samples),
            cpu_topk_max_ns=max(samples),cpu_topk_samples_ns=json.dumps(samples),
            cpu_timing_source='Actual CPU Measurement; all four query heads; warmup excluded'))
        result['cpu_functional_benchmark'].append(dict(common,mode=mode,
            cpu_topk_median_ns=float(np.median(samples)),cpu_topk_min_ns=min(samples),
            cpu_topk_max_ns=max(samples),cpu_topk_samples_ns=json.dumps(samples),
            encoding_cpu_ns=encode_ns,scoring_cpu_ns=score_cpu_ns,
            source='Actual CPU Measurement; four query heads; NOT GPU or PIM latency'))
        result['traffic_comparison'].append(dict(common,mode=mode,source='Analytical Model; logical payload bytes',
            **{k:v for k,v in cost.items() if 'bytes' in k}))
        result['hardware_cost'].append(dict(common,mode=mode,source='Analytical Model; logical complexity only',
            **{k:v for k,v in cost.items() if any(x in k for x in ['area','energy','operators','units','buffer','lanes','staging','candidate'])}))
    bank_ids = map_banks(context,banks,mapping)
    for h,s in enumerate(scores):
        for b in range(banks):
            ids = np.flatnonzero(bank_ids == b)
            gold_count = int(np.sum(bank_ids[refs[h]] == b))
            result['bank_distribution'].append(dict(common,head=h,bank=b,token_count=len(ids),
                score_mean=float(np.mean(s[ids].astype(float))) if len(ids) else None,
                score_std=float(np.std(s[ids].astype(float))) if len(ids) else None,
                score_min=float(np.min(s[ids])) if len(ids) else None,
                score_max=float(np.max(s[ids])) if len(ids) else None,
                global_topk_count=gold_count,global_topk_share=gold_count/budget,
                fixed_quota=(int(selected_by_mode['bank_local'][h]['quotas'][b]) if 'bank_local' in modes else None),
                adaptive_quota=(int(selected_by_mode['bank_adaptive'][h]['quotas'][b]) if 'bank_adaptive' in modes else None),
                source='Actual CPU Measurement; synthetic score distribution'))
    config = dict(common,experiment_type='Synthetic Experiment',q_heads=4,kv_heads=2,batch=1,
        dim=128,dim_group=64,token_group=64,input_dtype='float16',score_dtype='float16',
        tie_breaking='score descending, token index ascending',chunk_size=chunk_size,repeats=repeats,
        current_token_contract='cached-prefix selection only; budget slots; current token not included',
        bank_bias=('iid Gaussian' if scenario == 'iid' else 'first 1/8 banks: K += 3*first mapped Q'),
        hardware_assumptions=asdict(hw),environment=environment(),
        original_commit='623349ca4cb9ab77029201595acb688d60d32946',
        source='CPU functional experiment + uncalibrated analytical model; no CUDA/RTL/DRAM timing')
    return result,config,costs,scores


def save_case(output,result,config,costs):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        write_csv(output/(name+'.csv'),rows)
    (output/'experiment_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    write_trace(output/'analytical_trace.csv',costs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context-length',type=int,default=4096)
    p.add_argument('--budget',type=int,default=512)
    p.add_argument('--num-banks',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scenario',choices=['iid','bank_cluster'],default='iid')
    p.add_argument('--mapping',choices=['striped','contiguous'],default='striped')
    p.add_argument('--modes',default=','.join(MODES))
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--chunk-size',type=int,default=1024)
    p.add_argument('--hardware-config',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/functional'))
    args = p.parse_args()
    hw = HardwareConfig(**json.loads(args.hardware_config.read_text())) if args.hardware_config else None
    result,config,costs,_ = run_case(args.context_length,args.budget,args.num_banks,args.seed,
        args.scenario,args.mapping,tuple(args.modes.split(',')),args.repeats,args.chunk_size,hw)
    save_case(args.output_dir,result,config,costs)
    print(json.dumps({'output':str(args.output_dir),'score_sha256':config['score_sha256'],
                      'all_exact_modes_verified':True},indent=2))


if __name__ == '__main__': main()
