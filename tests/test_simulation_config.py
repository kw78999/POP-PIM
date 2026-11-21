"""The configuration builder reproduces the exact YAML of the archived development runs (paths normalized)."""
import json
from pathlib import Path

import pytest

from tools.common import ROOT
from tools.run_simulation import BANDWIDTHS, VARIANTS, Case, configuration, grid, host_topk_ps, resolve_variant

FIXTURES = Path(__file__).parent / 'fixtures/configs'
CASES = json.loads((FIXTURES / 'cases.json').read_text())


@pytest.mark.parametrize('name', sorted(CASES))
def test_configuration_matches_archived_run(name):
    spec = CASES[name]
    case = Case(spec['design'], spec['variant'], spec['bandwidth'], spec['refresh'], spec['context'], spec['link_GBs'])
    assert case.name == name
    folder = Path('/OUT') / name
    text = configuration(case, folder, replay_dir=Path('/REPLAY'))
    text = text.replace(str(folder), '<OUT>').replace('/REPLAY', '<REPLAY>').replace(str(ROOT), '<ROOT>')
    assert text == (FIXTURES / f'{name}.yaml').read_text()


def test_named_variants_and_host_topk():
    assert resolve_variant('POP-PIM-S') == 'gs4096_vs005h_ct'
    assert resolve_variant('POP-PIM-Q') == 'gs6144_vs002h_ct'
    assert host_topk_ps('POP_S_GPU', 'base') == 200_000_000
    assert host_topk_ps('POP_S_CPU', 'base') == 2_786_760_000          # 32 rows x Top-4096
    assert host_topk_ps('POP_S_CPU', 'gs4096_vs005h_ct') == 778_347_000  # 8 rows x Top-4096
    assert host_topk_ps('POP_S_CPU', 'gs6144_vs002h_ct') == 1_047_870_000
    assert host_topk_ps('POP_S_CPU', 'base', override_ms=1.5) == 1_500_000_000


def test_invalid_cases_are_rejected():
    with pytest.raises(ValueError):
        Case('POP_E24P32', 'gs4096_ct').validate()       # compact transport needs the S path
    with pytest.raises(ValueError):
        Case('POP_S_CPU', context=65536).validate()      # fixed 32K replay
    with pytest.raises(ValueError):
        Case('D8', 'gs4096').validate()                  # variants are POP-only
    with pytest.raises(ValueError):
        Case('D8', bandwidth=30.0).validate()
    with pytest.raises(ValueError):
        Case('D8', link_GBs=8.0).validate()


def test_grid_skips_compact_transport_on_local_selector():
    cases = grid(['POP_E24P32', 'POP_S_GPU'], list(VARIANTS), sorted(BANDWIDTHS))
    assert not any(c.design == 'POP_E24P32' and VARIANTS[c.variant][3] for c in cases)
    assert len({c.name for c in cases}) == len(cases) == (9 + 14) * 4
