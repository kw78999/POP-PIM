# POP-PIM 한국어 안내

POP-PIM은 긴 문맥 디코딩에서 매 토큰마다 KV 캐시 전체를 다시 읽는 비용을 줄이기 위해, popcount 기반 근사
점수를 HBM bank 옆에서 계산하고 선택된 토큰의 원래 정밀도 K/V만 읽어 정확한 attention을 수행하는 구조입니다.
이 저장소는 그 하드웨어 배치·데이터 흐름·비용을 재현 가능한 형태로 공개합니다. 검색 수식은
[PopKV](https://github.com/kw78999/PopKV)를 따릅니다.

## 무엇을 비교하나

- **Dense D4/D8/D16/D32**: bank당 BF16 FMA lane 수(D#)를 늘린 근거리 메모리 dense attention. layer·token당
  128 MiB의 K/V를 모두 읽습니다.
- **POP-PIM**: query head마다 Top-4,096을 고르는 기본 구조(head 4개가 공유하는 KV head에서는 합집합을 읽음).
- **POP-PIM-S / POP-PIM-Q**: 바이트 절감 세 가지를 더한 구조입니다.
  - **GS**: 같은 KV head를 쓰는 4개 head의 BF16 근사 점수를 `((s0+s1)+(s2+s3))×0.25`로 합쳐 KV head당 Top-K를 한 번만 고릅니다.
  - **VS**: 선택된 K로 정확한 softmax를 먼저 구하고, 각 head 선택 질량의 1−δ를 유지하는 데 필요한 토큰의 V만 읽습니다(하드웨어 규칙은 정확한 규칙의 상위집합).
  - **CT**: 호스트로는 KV head·토큰당 위치 암묵적 2-byte 점수 하나만 보내고, index는 N-bit bitmap으로 돌려받습니다.
  - POP-PIM-S = GS(K=4,096) + 하드웨어 VS(δ=0.005) + CT, POP-PIM-Q = GS(K=6,144) + 하드웨어 VS(δ=0.002) + CT.

조건은 Llama-3.1-8B 한 layer의 decode 1 token(32 query head / 8 KV head, d=128), 문맥 N=32,768, HBM3 한 채널
(64 bank, 41.6 GB/s, all-bank 대역폭 증폭 없음), PIM-aware per-bank refresh입니다. 채널 대역폭은 41.6/20.8/10.4/5.2
GB/s로 바꿔 봅니다. 대표 표의 POP-PIM 계열은 호스트(CPU 8-thread, 실측) Top-k를 쓰는 S 경로입니다.

결과 표와 그림은 [README](../README.md#results)에 있고, 해석할 때의 범위·한계는 [results.md](results.md)에
정리했습니다. 지연은 한 layer·한 token의 시뮬레이션 attention 시간이며, 면적은 알려진 부분 면적(known partial
area)이지 전체 칩 면적이 아닙니다. 에너지는 측정하지 않았습니다.

## 가볍게 돌려보기

**1단계: CPU만으로 확인 (약 1분, 컴파일러 불필요)**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m tools.rebuild_tables --suite main --output-dir outputs/tables/main
cat outputs/tables/main/main.md
```

테스트는 기능 모델과 BF16 oracle, GS 입력·V-keep 마스크 해시, 설정 생성기, 면적 모델, 커밋된 참조 결과의 불변식을
확인합니다. `rebuild_tables`는 커밋된 실행 기록으로 대표 표와 그림을 다시 만듭니다(시뮬레이션은 하지 않음).

**2단계: 시뮬레이터 빌드 후 3개 case 실행 (빌드 약 5분 + 실행 약 1분)**

```bash
sudo apt-get install -y build-essential g++-10 cmake verilator   # Ubuntu 20.04/22.04
export CXX=g++-10                 # 참조 결과를 만든 컴파일러
python -m tools.build --jobs 4
python -m tools.reproduce --suite quick
```

`quick`은 41.6 GB/s에서 Dense D8, POP-PIM, POP-PIM-S를 새로 시뮬레이션하고, 각 raw command trace를 감사한 뒤,
`results/main/runs.json`의 참조값과 모든 지표를 비교합니다. 마지막에 "All 3 cases are identical"이 나오면 재현된
것입니다. 결과 표는 `outputs/quick/tables/quick.md`에 생깁니다.

**3단계: 전체 재현 (4코어 기준 약 1시간)**

```bash
python -m tools.reproduce --suite main        # 28 case, 대표 표
python -m tools.reproduce --suite ablation    # 121 case, GS/VS/CT ablation, host link sweep
python -m tools.reproduce --suite refresh     # 45 case, refresh 정책 비교
```

원하는 조건만 따로 돌릴 수도 있습니다.

```bash
python -m tools.run_simulation --designs D8 POP_S_CPU --variants POP-PIM-S \
  --bandwidths 41.6 5.2 --audit --output-dir outputs/demo
```

RTL 검증(`tools.verify_rtl`), 합성·CACTI 재현, 호스트 Top-k 재측정, 선택 실험(모델 품질)은
[재현 가이드](reproducing.md)에 단계별로 있습니다.

## 저장소 구성과 분리한 자료

코드 위치와 역할은 [architecture.md](architecture.md)에 있습니다. 개발 중의 탐색 실험, 디버그 스크립트, 보고서와
원고, 대용량 raw 결과는 이 저장소에 넣지 않았고, 어떤 단계의 무엇을 남기고 분리했는지는 [history.md](history.md)에
정리했습니다. 공개 코드는 분리한 자료에 의존하지 않습니다.

프로젝트 라이선스와 논문 저자·제목·DOI는 아직 정해지지 않아 기재하지 않았습니다. 게시 전에 채울 항목은
[publishing.md](publishing.md)에 있습니다.
