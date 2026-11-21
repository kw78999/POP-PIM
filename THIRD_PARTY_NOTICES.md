# Third-party code and models

| Component | Location and license | Provenance |
|---|---|---|
| Ramulator 2 | `third_party/ramulator/LICENSE`, MIT | Base commit and local tracked diff in `third_party/versions.json` and `upstream_tracked_changes.patch` |
| AttAcc modifications | `third_party/ramulator/LICENSE.AttAcc`, MIT | AttAcc commit `c60005143a6b492d7ef83231723386478b59a506` |
| yaml-cpp | `third_party/ramulator/ext/yaml-cpp/LICENSE`, MIT | Exact commit recorded in `third_party/versions.json` |
| spdlog and bundled fmt | `third_party/ramulator/ext/spdlog/LICENSE` and notices in its headers | Exact commit recorded in `third_party/versions.json` |
| argparse | `third_party/ramulator/ext/argparse/LICENSE`, MIT | Exact commit recorded in `third_party/versions.json` |
| Berkeley HardFloat Release 1 | `third_party/hardfloat/COPYING.txt`, BSD 3-clause | Source notices retained |
| Nangate45 Liberty | Downloaded separately; not distributed here | URL and SHA-256 recorded in `third_party/versions.json`; follow the upstream terms |
| CACTI | Optional external source/tool; retained estimator inputs and outputs under `data/area/` | Revision and source hashes recorded; `tools.reproduce_cacti --fetch` explicitly retrieves that revision |

The HBM device model and controller under `simulator/common/` derive from Ramulator/AttAcc code; the corresponding MIT notices above apply to those derived portions. The custom CMake file uses local copies in place of upstream FetchContent downloads. Vendored source, including locally added files, is authoritative; the tracked diff alone is not a complete replacement for that source.

PopKV's sign/magnitude encoding and score formula are research provenance, not a new mathematical contribution of this repository. See the [PopKV source](https://github.com/kw78999/PopKV). The included replay contains score bits and token indices; the V-keep masks (`data/replay/variants/`) and VS test vectors (`data/rtl_vectors/vs/`) add selection bits and FP32 attention logits of the same single Llama-3.1-8B decode step. None of them contains model weights, token IDs or corpus text. The optional quality evaluation downloads Llama-3.1-8B and reads PG19 under their own licenses; neither is distributed here.
