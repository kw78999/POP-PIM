"""Report tables rebuilt from the committed records match the committed tables."""
import csv

import pytest

from tools.common import ROOT
from tools.rebuild_tables import rebuild


@pytest.mark.parametrize('suite', ['main', 'ablation', 'refresh'])
def test_rebuilt_tables_match(tmp_path, suite):
    rebuild(suite, tmp_path, figures=False)
    committed = ROOT / 'results' / suite
    assert (tmp_path / f'{suite}.md').read_text() == (committed / f'{suite}.md').read_text()
    with (tmp_path / f'{suite}.csv').open() as a, (committed / f'{suite}.csv').open() as b:
        assert list(csv.reader(a)) == list(csv.reader(b))
