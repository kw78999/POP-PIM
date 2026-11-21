"""Download only the pinned Liberty file, verifying its complete SHA-256."""
import json
import urllib.request
from .common import ROOT, LIB, sha


def main():
    metadata = json.loads((ROOT/'third_party/versions.json').read_text())['nangate45']
    if LIB.exists():
        if sha(LIB) != metadata['sha256']:
            raise ValueError(f'Existing Liberty file differs from the pinned reference: {LIB}')
        print(f'Already verified: {LIB}')
        return
    LIB.parent.mkdir(parents=True, exist_ok=True)
    temporary = LIB.with_suffix('.download')
    urllib.request.urlretrieve(metadata['url'], temporary)
    if sha(temporary) != metadata['sha256']:
        temporary.unlink()
        raise ValueError('Liberty download hash mismatch')
    temporary.replace(LIB)
    print(f'Downloaded and verified: {LIB}')


if __name__ == '__main__':
    main()
