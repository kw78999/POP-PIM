"""Regenerate POP-PIM replay variants (GS scores/indices, unpacked V-keep masks) and verify their hashes.

Inputs are written under ``<build>/replay`` (``POPPIM_BUILD_DIR`` or ``build/``). Existing files are
reused only if their SHA-256 still matches ``data/replay/variants/manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from poppim.variants import QUERY_HEADS, replay_variant, union_tokens, unpack_mask
from .common import BUILD, ROOT

VARIANTS_DIR = ROOT / 'data/replay/variants'
MANIFEST = VARIANTS_DIR / 'manifest.json'
BASE_SCORES = ROOT / 'data/replay/scores_bf16.bin'
BASE_INDICES = ROOT / 'data/replay/topk_indices_i32.bin'


def manifest():
    return json.loads(MANIFEST.read_text())


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_checked(path, data, expected):
    if _digest(data) != expected:
        raise RuntimeError(f'{path.name}: regenerated content does not match the release manifest')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or _digest(path.read_bytes()) != expected:
        path.write_bytes(data)


def _valid(path, expected):
    return path.is_file() and _digest(path.read_bytes()) == expected


def input_paths(name, output_dir=None):
    """``(scores, expected)`` paths for an input variant; the base variant uses the committed replay."""
    if name == 'base':
        return BASE_SCORES, BASE_INDICES
    folder = Path(output_dir or BUILD / 'replay') / name
    return folder / 'scores.bin', folder / 'expected.bin'


def mask_path(key, output_dir=None):
    return Path(output_dir or BUILD / 'replay') / 'masks' / f'vkeep_{key}.bin'


def prepare(output_dir=None, quiet=False):
    """Create every variant input and mask; return a summary with paths and hashes."""
    spec = manifest()
    n = spec['N']
    base = np.fromfile(BASE_SCORES, dtype='<u2').reshape(QUERY_HEADS, n)
    summary = dict(inputs={}, vkeep={})
    for name, item in spec['inputs'].items():
        scores_path, expected_path = input_paths(name, output_dir)
        if name == 'base' or not (_valid(scores_path, item['scores_sha256']) and _valid(expected_path, item['expected_sha256'])):
            scores, expected = replay_variant(base, item['budget'], shared=name != 'base')
            if union_tokens(expected) != item['union_tokens_per_kv']:
                raise RuntimeError(f'{name}: selected union differs from the manifest')
            if name != 'base':
                _write_checked(scores_path, scores.astype('<u2').tobytes(), item['scores_sha256'])
                _write_checked(expected_path, expected.astype('<i4').tobytes(), item['expected_sha256'])
        for path, key in ((scores_path, 'scores_sha256'), (expected_path, 'expected_sha256')):
            if not _valid(path, item[key]):
                raise RuntimeError(f'{path}: hash mismatch')
        summary['inputs'][name] = dict(scores=str(scores_path), expected=str(expected_path), budget=item['budget'])
    for key, item in spec['vkeep'].items():
        path = mask_path(key, output_dir)
        if not _valid(path, item['unpacked_sha256']):
            packed = (VARIANTS_DIR / item['file']).read_bytes()
            if _digest(packed) != item['packed_sha256']:
                raise RuntimeError(f'{item["file"]}: packed mask hash mismatch')
            mask = unpack_mask(packed, n)
            if int(mask.sum()) != item['kept_v_tokens']:
                raise RuntimeError(f'{key}: kept-token count mismatch')
            _write_checked(path, mask.astype(np.uint8).tobytes(), item['unpacked_sha256'])
        summary['vkeep'][key] = str(path)
    if not quiet:
        print(f'Prepared {len(summary["inputs"])} replay inputs and {len(summary["vkeep"])} V-keep masks under '
              f'{Path(output_dir or BUILD / "replay")}', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='default: <build>/replay')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
