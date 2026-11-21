"""GS replay variants and V-keep masks regenerate the committed release inputs bit-exactly."""
import hashlib
import json

import numpy as np
import pytest

from poppim.variants import (bf16_to_f32, f32_to_bf16, gs_aggregate, replay_variant, stable_topk, union_tokens,
                             unpack_mask, vs_scan_cycles)
from tools.common import ROOT

MANIFEST = json.loads((ROOT / 'data/replay/variants/manifest.json').read_text())
N = MANIFEST['N']


@pytest.fixture(scope='module')
def base():
    return np.fromfile(ROOT / 'data/replay/scores_bf16.bin', dtype='<u2').reshape(32, N)


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.mark.parametrize('name', ['base', 'gs4096', 'gs6144'])
def test_replay_variant_hashes(base, name):
    spec = MANIFEST['inputs'][name]
    scores, expected = replay_variant(base, spec['budget'], shared=name != 'base')
    assert sha(scores.astype('<u2')) == spec['scores_sha256']
    assert sha(expected.astype('<i4')) == spec['expected_sha256']
    assert union_tokens(expected) == spec['union_tokens_per_kv']


def test_gs_aggregate_is_the_rounded_mean():
    bits = f32_to_bf16(np.array([[1.0], [2.0], [3.0], [4.0]] * 8, dtype=np.float32))
    assert np.all(bf16_to_f32(gs_aggregate(bits)) == np.float32(2.5))


def test_stable_topk_ties_prefer_lower_index():
    values = np.array([[1.0, 2.0, 2.0, 0.0, -0.0, 2.0]], dtype=np.float32)
    assert stable_topk(values, 2).tolist() == [[1, 2]]
    assert stable_topk(values, 5).tolist() == [[0, 1, 2, 3, 5]]


@pytest.mark.parametrize('key', sorted(MANIFEST['vkeep']))
def test_masks_unpack_to_recorded_hash(key):
    spec = MANIFEST['vkeep'][key]
    mask = unpack_mask((ROOT / 'data/replay/variants' / spec['file']).read_bytes(), N)
    assert int(mask.sum()) == spec['kept_v_tokens']
    assert hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest() == spec['unpacked_sha256']


def test_hardware_masks_contain_the_exact_masks():
    for key, spec in MANIFEST['vkeep'].items():
        if key.endswith('_hw'):
            exact = MANIFEST['vkeep'][key[:-3] + '_exact']
            hw = unpack_mask((ROOT / 'data/replay/variants' / spec['file']).read_bytes(), N)
            ex = unpack_mask((ROOT / 'data/replay/variants' / exact['file']).read_bytes(), N)
            assert spec['superset_of_exact'] and np.all(hw >= ex)


def test_vs_scan_cycles():
    # POP-PIM-S (GS Ks=4096, hardware VS delta=0.005): 1628 cycles = 13.0 us at 125 MHz.
    assert vs_scan_cycles(MANIFEST['vkeep']['gs4096_0.005_hw']['bstar']) == 1628
    with pytest.raises(ValueError):
        vs_scan_cycles([0] * 8)
