import csv
import importlib.util
from pathlib import Path
import numpy as np
from poppim.simulator.trace_generator import write_trace
from poppim.simulator.cost_model import estimate
from poppim.topk.near_memory_topk import StreamingTopK


def test_streaming_buffer_and_reproducible_scores():
    spec = importlib.util.spec_from_file_location('run_comparison',
                Path(__file__).parents[1]/'experiments/run_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first,config,_,scores = module.run_case(64,8,16,repeats=1)
    _,other,_,again = module.run_case(64,8,16,repeats=1)
    assert config['score_sha256'] == other['score_sha256']
    np.testing.assert_array_equal(scores,again)
    selector = StreamingTopK(8)
    for start in range(0,64,7):
        selector.feed(scores[0,start:start+7],np.arange(start,min(64,start+7)))
        assert len(selector.heap) <= 8
    assert all(r['exact_match'] == 1 for r in first['topk_accuracy'] if r['mode'] in ('host','near_memory'))


def test_trace_modes(tmp_path):
    path = tmp_path/'trace.csv'
    costs = [estimate(128,16,4),estimate(128,16,4,mode='near_memory')]
    write_trace(path,costs)
    rows = list(csv.DictReader(path.open()))
    assert not any('LOCAL_TOPK' in r['command'] for r in rows)
    assert any(r['command']=='PIM_RETURN_SCORE' and r['mode']=='host' for r in rows)
    assert any('NEAR_MEMORY_GLOBAL_TOPK' in r['command'] for r in rows)
    for cost in costs:
        serial = sum(float(r['duration_ns']) for r in rows if r['mode']==cost['mode'] and r['overlap_group']=='serial')
        stream = max([float(r['duration_ns']) for r in rows if r['mode']==cost['mode'] and r['overlap_group']=='stream'] or [0])
        assert abs(serial+stream-cost['total_retrieval_ns']) < 1e-6
