"""Check the portable release contents and their SHA-256 manifest without building anything."""
import argparse
import json
import os
import re
from pathlib import Path
from .common import ROOT, save, sha

EXCLUDE_DIRS = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', 'outputs', 'downloads',
                '.idea', '.vscode', 'research-history', 'nangate45'}
EXCLUDE_SUFFIXES = {'.pyc', '.pyo', '.so', '.a', '.o', '.vcd', '.fst', '.zip', '.gz'}
MANIFEST = ROOT / 'provenance/release_manifest.json'
REQUIRED = ['README.md', 'LICENSE.md', 'CITATION.md', 'CONTRIBUTING.md', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md',
            'docs/reproducing.md', 'docs/results.md', 'docs/architecture.md', 'docs/guide_ko.md',
            'data/replay/manifest.json', 'data/replay/variants/manifest.json', 'data/area/extension_components.json',
            'data/calibration/host_topk.json', 'results/main/runs.json', 'results/ablation/runs.json',
            'results/refresh/runs.json', 'results/quality/quality_ppl.csv', '.github/workflows/ci.yml']
SOURCE_DIRS = {'tools', 'poppim', 'simulator', 'tests', 'experiments', 'rtl', 'docs', 'configs'}
SOURCE_SUFFIXES = ('.py', '.sh', '.cpp', '.h', '.inc', '.sv', '.md', '.yaml', '.yml', '.toml')
# Development-only material that must not reappear in the release tree.
LOCAL_ROOT = '/' + 'workspace' + '/'   # development machine prefix, spelled so this file does not match
FORBIDDEN = re.compile(r'(^|/)(build_phase\d+|phase\d+)(/|$)')


def public_files():
    files = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.venv') and not d.endswith('.egg-info'))
        for name in sorted(names):
            path = Path(directory) / name
            if path == MANIFEST or path.suffix in EXCLUDE_SUFFIXES or name.startswith('.env') or name == '.DS_Store':
                continue
            if path.is_symlink():
                raise ValueError(f'Unexpected release symlink: {path.relative_to(ROOT)}')
            files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Intentionally record the reviewed current release contents')
    args = parser.parse_args()
    files = public_files()
    current = {str(path.relative_to(ROOT)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in files}
    large = [name for name, info in current.items() if info['bytes'] >= 50 * 1024 * 1024]
    assert not large, f'Large release files; publish them as separate archived artifacts: {large}'
    for required in REQUIRED:
        assert required in current, f'Missing release component: {required}'
    leftovers = [name for name in current if FORBIDDEN.search(name) and not name.startswith('data/area/cacti/')]
    assert not leftovers, f'Development phase material in the release tree: {leftovers[:5]}'
    for name in current:
        top = name.split('/')[0]
        if (top in SOURCE_DIRS or '/' not in name) and name.endswith(SOURCE_SUFFIXES):
            text = (ROOT / name).read_text(errors='replace')
            assert LOCAL_ROOT not in text, f'Nonportable absolute path in {name}'
    if args.write_manifest:
        save(MANIFEST, dict(format_version=1, files=current,
             scope='Source, small replay inputs, compact reference evidence and documentation; generated outputs excluded'))
    else:
        manifest = json.loads(MANIFEST.read_text())
        assert manifest['files'] == current, 'Release changed; review the diff and intentionally regenerate the manifest'
    print(json.dumps(dict(status='passed', files=len(files), total_bytes=sum(info['bytes'] for info in current.values()),
                          project_license='pending maintainer decision', publication_metadata='pending maintainer input'), indent=2))


if __name__ == '__main__':
    main()
