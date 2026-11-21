"""Invariants of the committed reference records (metrics-level audit; the raw traces were audited when produced)."""
import json

import pytest

from tools.audit_simulation import audit_work
from tools.common import ROOT
from tools.suites import REFERENCE, cases

SUITES = ['main', 'ablation', 'refresh']


def records(suite):
    return json.loads((ROOT / 'results' / suite / 'runs.json').read_text())


@pytest.mark.parametrize('suite', SUITES)
def test_reference_covers_exactly_the_suite(suite):
    assert sorted(r['name'] for r in records(suite)) == sorted(c.name for c in cases(suite))


@pytest.mark.parametrize('suite', SUITES)
def test_reference_records_pass_metric_audit(suite):
    for record in records(suite):
        audit_work(record, record['metrics'])
        assert record['audit']['passed'] and record['audit']['name'] == record['name']
        if record['refresh'].startswith('per-bank'):
            assert record['audit']['REFsb'] == record['metrics']['REFsb'] > 0
            assert record['audit']['max_debt'] <= record['audit']['postpone']


def test_quick_suite_is_part_of_main():
    names = {r['name'] for r in records(REFERENCE['quick'])}
    assert {c.name for c in cases('quick')} <= names


def test_all_bank_rows_reproduce_the_v0_1_comparison():
    old = {r['design']: r['metrics'] for r in json.loads((ROOT / 'results/reference/all_bank_refresh_v0.1/runs.json').read_text())}
    new = {r['design']: r['metrics'] for r in records('refresh') if r['refresh'] == 'all-bank'}
    for design, metrics in old.items():
        shared = set(metrics) & set(new[design])
        assert len(shared) > 40
        assert {k: metrics[k] for k in shared} == {k: new[design][k] for k in shared}, design
