"""Known-partial area model: recorded component values and design/variant totals."""
import pytest

from tools.area_model import area, sram


EXACT, ROUNDED = 1e-12, 5e-4   # full-precision recorded values / values recorded to three decimals


@pytest.mark.parametrize('design, variant, refresh, expected, tol', [
    ('D4', 'base', 'per-bank', 4.994392362896025, EXACT),
    ('D8', 'base', 'per-bank', 6.584672298896154, EXACT),
    ('D16', 'base', 'per-bank', 9.954658218892805, EXACT),
    ('D32', 'base', 'per-bank', 16.53957546690156, EXACT),
    ('D1', 'base', 'all-bank', 3.775841840896032, EXACT),          # no refresh scheduler with REFab
    ('POP_E24P32', 'base', 'per-bank', 24.5273520655815, EXACT),
    ('POP_S_CPU', 'base', 'per-bank', 9.652, ROUNDED),             # POP-PIM baseline incl. 4-bit membership store
    ('POP_S_CPU', 'gs4096_vs005h_ct', 'per-bank', 11.008, ROUNDED),  # POP-PIM-S
    ('POP_S_CPU', 'gs6144_vs002h_ct', 'per-bank', 11.892, ROUNDED),  # POP-PIM-Q
    ('POP_E24P32', 'gs4096', 'per-bank', 13.578, ROUNDED),         # GS shrinks the local selector SRAM
])
def test_known_partial_area(design, variant, refresh, expected, tol):
    assert area(design, variant, refresh)['known_partial_mm2'] == pytest.approx(expected, abs=tol)


def test_components_sum():
    parts = area('POP_S_CPU', 'gs4096_vs005h_ct')
    total = parts.pop('known_partial_mm2')
    assert total == pytest.approx(sum(parts.values()), rel=1e-15)
    assert parts['staging_delta_mm2'] < 0                     # 16-bit packets shrink the staging FIFOs
    assert sram(65536) == pytest.approx(0.39295, abs=1e-5)


def test_local_selector_gs_model_limited_to_e24p32():
    with pytest.raises(ValueError):
        area('POP_E4P4', 'gs4096')
