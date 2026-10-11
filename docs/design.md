# servcalc 설계 문서

작성일: 2026-10-11
상태: 1차 리뷰 반영

## 1. 목적

Hugging Face `config.json`과 GPU 사양만으로 단일 GPU에서의 LLM 서빙 용량을 예측하는 Python 라이브러리입니다. vramcalc가 "이 모델이 이 GPU에 들어가는가"를 답한다면, servcalc는 그 다음 질문인 "동시에 몇 개의 시퀀스를 처리할 수 있고, 초당 몇 토큰이 나오며, 첫 토큰과 토큰 간 지연은 얼마인가"를 배치 크기의 함수로 답합니다.

핵심은 roofline 모델입니다. decode step은 매번 가중치와 활성 시퀀스의 KV cache를 읽으므로 메모리 대역폭에 묶이고, prefill은 연산량에 묶입니다. 작업량(바이트, FLOPs)은 모델 구조와 스케줄링 정책으로 결정됩니다. 작업량을 시간으로 바꾸는 효율 계수만 엔진, 배치 크기, 양자화에 따라 다르고, 이 부분은 실측으로 보정합니다.

### 1.1 성공 기준

- 작업량 식(5절)이 tiny 아키텍처 손계산과 정확히 일치합니다.
- Llama-3.1-8B + RTX 3090 + vLLM 같은 대표 조합에서 결과가 상식 범위(예: bf16 8B의 N=1 ITL 15~40ms)에 들어옵니다.
- 세 엔진 프로필(`vllm`, `ollama`, `hf`)로 같은 API에서 "ollama Q4로 7B를 3090에서 동시 4명이면 각 몇 tok/s, vLLM bf16으로 바꾸면 몇 명까지"를 답할 수 있습니다.
- 모든 지표가 범위(low, mid, high)로 나오고, 실측 전에는 사전 범위임을 결과와 README에 명시합니다.
- PyPI에 배포되고 GitHub Actions CI가 통과합니다.

### 1.2 1차 범위 밖 (0.2 이후)

- 요청 도착률과 길이 분포를 받는 이벤트 시뮬레이터, 대기열 시간, 지연 시간 백분위
- speculative decoding, prefix caching
- tensor parallel(GPU 여러 장), CPU offload(ollama가 VRAM 부족 시 자동 수행), MoE, MLA
- 추가 엔진 프로필(SGLang, TGI, TensorRT-LLM, ExLlamaV2)
- 실측 보정. 1차는 사전 범위만 내장하고, 0.2에서 RunPod 실측으로 `measured` 곡선으로 교체
- 정확도 수치 약속. vramcalc의 ±4%는 메모리에 대한 것이며 시간 예측에는 적용되지 않습니다
- activation 트래픽, 양자화 포맷별 잔류 fp16 텐서의 정확한 바이트. 1차는 근사치를 쓰고 `weight_bytes` override로 보정할 수 있게 합니다

## 2. 설계 원칙

- **작업량과 시간을 분리합니다.** 바이트와 FLOPs는 결정적이고 테스트로 고정합니다. 시간 변환만 효율 곡선에 의존합니다.
- **스케줄링 정책을 명시합니다.** continuous 모드의 모든 지표는 "토큰 예산 B, FCFS, 가장 오래된 prefill 요청에 남은 예산을 몰아준다"는 하나의 정책에서 유도합니다. 정상 상태는 보존 법칙(들어오는 prefill 토큰 = 처리하는 prefill 토큰)을 만족해야 하고, 만족하지 못하는 N은 `feasible=False`입니다.
- **엔진 차이는 프로필 값으로 흡수합니다.** 배치 방식, 동시 처리 상한, 양자화 포맷, KV 관리, 메모리 예산, 토큰 예산, prefill과 decode의 혼합 여부, KV 재할당 여부, 보정 곡선. 코어 함수는 엔진 이름을 모릅니다.
- **지원하지 않는 설정은 거부합니다.** 조용히 틀린 숫자를 내지 않습니다.
- **동시성을 둘로 나눕니다.** VRAM이 허용하는 수(`memory_capacity_seqs`)와 엔진 설정 상한(`configured_capacity_seqs`). 상주 요청 수에는 decode 중인 요청과 prefill 중인 요청이 모두 포함됩니다.
- **저장 크기, 스트리밍 바이트, 실행 비용을 구분합니다.** 상주 가중치(용량), step마다 읽는 가중치(임베딩 lookup 제외), 양자화 커널 속도(보정 곡선)는 서로 다른 값입니다.
- **vramcalc를 재사용합니다.** `Arch`, `load_arch`, `memory.kv_cache`, GPU 이름 해석을 import합니다. 의존 방향은 `servcalc → vramcalc`입니다.

## 3. 패키지 구조

```
servcalc/
├── pyproject.toml              # dependencies = ["vramcalc>=0.1.1"]
├── README.md                   # 영어 + 한국어
├── CHANGELOG.md
├── LICENSE                     # MIT
├── .github/workflows/ci.yml
├── docs/design.md
├── src/servcalc/
│   ├── __init__.py
│   ├── gpus.py                 # GpuSpec, vramcalc 테이블 + bandwidth_gbs, bf16_tflops
│   ├── workload.py             # Workload
│   ├── quant.py                # 양자화 포맷 테이블 (비트, 보정 곡선 이름)
│   ├── profiles.py             # EngineProfile, vllm(), ollama(), hf()
│   ├── calibration.py          # Curve, Calibration, 내장 사전 곡선
│   ├── work.py                 # 작업량 순수 함수
│   ├── timing.py               # 작업량 + 곡선 → 시간, bound, Range
│   ├── modes.py                # static, continuous 정상 상태 식, 용량
│   ├── estimate.py             # estimate(), ServingEstimate, BatchEstimate
│   └── cli.py
└── tests/
```

### 3.1 의존성

기본 설치: `vramcalc>=0.1.1` 하나. `servcalc[hf]`는 `vramcalc[hf]`를 그대로 노출합니다. Python `>=3.10`, uv, hatchling, ruff, pytest.

## 4. 공개 API

### 4.1 입력

```python
@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_bytes: int
    bandwidth_gbs: float           # 메모리 대역폭 GB/s (제조사 스펙)
    bf16_tflops: float             # dense bf16 피크 TFLOPs (sparsity 수치 제외)

@dataclass(frozen=True)
class Workload:
    prompt_len: int                # I
    gen_len: int                   # O ≥ 2. 첫 토큰은 prefill이 내므로 decode step은 D = O − 1. prefill 전용(O = 1)은 1차 범위 밖
    concurrency: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

@dataclass(frozen=True)
class EngineProfile:
    name: str
    version: str                   # 기본값이 기준한 엔진 버전 (patch까지)
    batching: str                  # "continuous" | "static"
    max_seqs: int | None           # 엔진 상주 요청 상한. None이면 무제한
    quant: str | None              # None(bf16) | "fp8" | "awq" | "gptq" | "gguf_q8_0" | "gguf_q4_k_m" | ...
    kv_bits: int                   # 16 | 8
    kv_block: int                  # paged 블록(토큰). 0이면 연속 할당
    kv_prealloc_ctx: int | None    # ollama: num_ctx. 시퀀스당 KV를 이 길이로 미리 할당
    memory_budget_ratio: float     # 엔진이 쓰기로 한 VRAM 비율 (vLLM gpu_memory_utilization)
    workspace_ratio: float         # 예산 안에서 activation, cuBLAS, 단편화가 쓰는 비율 (추정치)
    token_budget: int | None       # step당 총 토큰 예산 B (decode 포함). None이면 prefill 통째
    mixes_prefill: bool            # True면 한 step에 prefill과 decode가 섞임, False면 prefill이 별도 pass
    kv_realloc: bool               # True면 step마다 KV 전체를 복사 (HF DynamicCache)
    calibration: str               # 내장 보정 곡선 이름

@dataclass(frozen=True)
class Curve:
    points: tuple[tuple[float, float, float, float], ...]   # (x, low, mid, high), x 오름차순, 구간별 선형 보간, 밖은 clamp

@dataclass(frozen=True)
class Calibration:
    bw_util: Curve                 # x = step의 총 토큰 수 → 대역폭 활용률
    mfu: Curve                     # x = step의 총 토큰 수 → MFU
    overhead_ms: tuple[float, float, float]
    source: str                    # "prior" | "measured: <GPU> <engine> <version>"
```

### 4.2 진입점

```python
def estimate(
    model,                                   # dict | 경로 | Hub id (vramcalc.load_arch)
    gpu,                                     # 이름(str) | GpuSpec
    *,
    workload: Workload,
    profile: EngineProfile | str = "vllm",
    calibration: Calibration | None = None,  # None이면 profile.calibration의 내장 곡선
    weight_bytes: int | None = None,         # 상주 가중치 바이트 override (GGUF 파일 크기 등)
) -> ServingEstimate
```

우선순위: 명시 인자(`weight_bytes`, `calibration`, 프로필 팩토리 인자) > config 추론(`quantization_config`) > 프로필 기본값.

### 4.3 결과

```python
@dataclass(frozen=True)
class Range:
    low: float
    mid: float
    high: float

@dataclass(frozen=True)
class BatchEstimate:
    batch: int                     # N, decode 중인 요청 수
    resident: int                  # R = N + prefill 중인 요청 수 M
    feasible: bool
    infeasible_reason: str | None  # "memory" | "max_seqs" | "token_budget" | None
    ttft_ms: Range | None          # 대기열 제외. token_budget 위반으로 prefill을 표현할 수 없으면 None
    itl_ms: Range
    seq_tok_s: Range               # 1000 / itl
    total_tok_s: Range             # N × seq_tok_s. decode 출력만 센 값(prefill이 내는 첫 토큰 제외)
    bound: str                     # "memory" | "compute" | "overhead" (mid 기준)

@dataclass(frozen=True)
class ServingEstimate:
    memory_capacity_seqs: int      # VRAM이 허용하는 상주 요청 수 (모두 I + O 길이일 때)
    configured_capacity_seqs: int  # min(memory_capacity_seqs, max_seqs)
    saturating_batch: int | None   # concurrency 목록에서 bound(mid)가 memory → compute로 처음 바뀌는 N. 없으면 None
    points: tuple[BatchEstimate, ...]
    weight_bytes: int              # 상주
    weight_stream_bytes: int       # step마다 읽는 양
    kv_bytes_per_token: int
    params: EstimateParams         # arch, gpu, workload, profile, calibration.source
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]

    def max_users(self, itl_ms: float) -> int    # feasible한 N 중 itl_ms.mid ≤ itl_ms 인 최대 N, 없으면 0
    def table(self) -> str
    def to_dict(self) -> dict
```

Range 방향: 시간 지표(`ttft_ms`, `itl_ms`)의 low는 효율 high와 overhead low에서 나옵니다. 처리량 지표는 시간의 역수이므로 `seq_tok_s = (1000 / itl.high, 1000 / itl.mid, 1000 / itl.low)`, `total_tok_s = N × seq_tok_s`(세 값 모두에 곱함)입니다. 모든 Range는 `low ≤ mid ≤ high`를 만족합니다.

## 5. 작업량 식 (`work.py`)

기호: `P` 전체 파라미터, `P_mm` 행렬곱 참여 파라미터(tie면 `P`, 아니면 `P − V×h`; 입력 임베딩은 lookup이라 제외), `P_body = P_mm − V×h`(lm_head도 제외), `P_lin = L × (attn + mlp)`(양자화 대상 linear), `L` 레이어, `q = num_heads × head_dim`, `kv = num_kv_heads × head_dim`, `w` 가중치 비트, `k` KV 비트, `N` decode 중 요청 수, `c_i` 요청 i의 현재 컨텍스트, `I` prompt_len, `O` gen_len(≥ 2), `D = O − 1` decode step 수, `c̄ = I + 1 + (D − 1) / 2` decode 중 평균 컨텍스트(step j의 컨텍스트가 `I + 1 + j`, 0 ≤ j < D).

### 5.1 바이트

```
W_res     = P_lin × w / 8 + (P − P_lin) × 2        # 상주. 임베딩, lm_head, norm은 16비트 유지로 근사 (weight_bytes 지정 시 그 값)
W_stream  = W_res − (0 if tie else V × h × 2)       # step마다 읽는 양. untied 입력 임베딩은 읽지 않음
KV_tok    = 2 × L × kv × k / 8                      # vramcalc.memory.kv_cache(arch, 1, 1, k / 8)
```

양자화 포맷별 잔류 fp16 텐서(AWQ는 임베딩과 lm_head, GGUF는 output을 Q6_K 등)의 정확한 바이트는 1차 범위 밖이며 위 근사를 쓰고 `warnings`에 표기합니다.

### 5.2 decode step (N개 요청이 토큰 하나씩 생성)

```
bytes = W_stream
      + KV_tok × Σ c_i                 # KV 읽기
      + KV_tok × N                     # 새 토큰 KV 쓰기
      + (kv_realloc ? 2 × KV_tok × Σ c_i : 0)     # HF DynamicCache: cat으로 전체 복사 (읽기 + 쓰기)
flops = 2 × P_mm × N + 4 × L × q × Σ c_i
```

가중치 읽기는 N과 무관하고, KV 읽기와 FLOPs는 N에 비례합니다.

### 5.3 prefill (요청 하나, I 토큰)

logits는 마지막 토큰만 계산합니다(vLLM, 최신 transformers 모두). 조각 크기 c로 k = ceil(I / c)개 조각에 나누어 처리하면:

```
flops_total     = 2 × P_body × I + 2 × V × h + 2 × L × q × I × (I + 1)     # attention 합은 조각 크기와 무관
kv_write_total  = KV_tok × I
kv_read_total   = KV_tok × Σ_{j=0}^{k−1} (j × c)  = KV_tok × c × k × (k − 1) / 2   # 각 조각이 이전 조각들의 KV를 다시 읽음
weight_reads    = k × W_stream − (k − 1) × 2 × V × h      # lm_head는 마지막 조각만 읽음. decode와 섞인 step에서는 W_stream을 한 번만 셈
```

조각 하나(크기 C, 이전 컨텍스트 c₀)의 FLOPs는 `2 × P_body × C + 4 × L × q × (C × c₀ + C × (C + 1) / 2)`이며, 마지막 조각에만 `2 × V × h`가 더해집니다. 통째 prefill은 `k = 1`, `c = I`, `kv_read_total = 0`.

### 5.4 KV 용량과 상주 요청 수

```
usable            = VRAM × memory_budget_ratio − W_res − VRAM × workspace_ratio
seq_kv_peak       = KV_tok × kv_prealloc_ctx                       (kv_prealloc_ctx가 있으면)
                  = KV_tok × roundup(I + O, kv_block)              (없으면. kv_block = 0이면 올림 없음)
memory_capacity   = floor(usable / seq_kv_peak)
configured_cap    = min(memory_capacity, max_seqs)                 (max_seqs가 None이면 memory_capacity)
```

`usable ≤ 0`이면 `DoesNotFitError`. `kv_prealloc_ctx < I + O`이면 `ValueError`(컨텍스트 초과. 잘림과 context shift는 모델링하지 않음).

## 6. 효율 곡선과 시간 변환 (`calibration.py`, `timing.py`)

### 6.1 GPU 사양 (`gpus.py`)

vramcalc 테이블의 각 GPU에 `bandwidth_gbs`와 `bf16_tflops`를 추가합니다(vramcalc 0.1.2에서 RTX A6000, RTX 6000 Ada, A800, H800 추가). T4와 V100은 bf16이 없어 fp16 tensor 피크를 쓰며 테이블 주석에 표기합니다. 예: RTX 3090 936 / 71, RTX 3090 Ti 1008 / 80, RTX 4090 1008 / 165, A6000 768 / 155, A100 80GB 2039 / 312, H100 80GB 3350 / 989, L4 300 / 121, L40S 864 / 362. 이름 해석은 `vramcalc.gpus.lookup`을 쓰고 두 열은 servcalc의 dict에서 가져옵니다. 문자열 이름 또는 `GpuSpec`만 받으며 숫자 GB는 `ValueError`로 거부합니다(대역폭과 TFLOPs 없이는 시간을 낼 수 없음).

### 6.2 시간 변환

step에 들어가는 총 토큰 수 `T = N + (그 step의 prefill 토큰)`을 두 곡선의 공통 독립 변수로 씁니다.

```
t_mem_ms   = bytes / (bandwidth_gbs × 1e9 × bw_util(T)) × 1000
t_comp_ms  = flops / (bf16_tflops × 1e12 × mfu(T)) × 1000
step_ms    = max(t_mem_ms, t_comp_ms) + overhead_ms
bound      = "overhead" if overhead_ms > max(t_mem, t_comp)
             else "memory" if t_mem > t_comp else "compute"
```

roofline의 `max`는 메모리와 연산이 완전히 겹친다는 가정이며 실제보다 빠른 쪽으로 틀립니다. 그 차이는 효율 곡선이 흡수합니다. 범위는 (low, mid, high) 세 번 계산합니다.

### 6.3 내장 사전 곡선

모두 `source="prior"`이며 실측 전 추정치입니다. 두 곡선의 x는 step 총 토큰 수입니다.

| 이름 | bw_util (T=1 / 8 / 64 / 512) | mfu (T=1 / 64 / 512 / 4096) | overhead_ms |
|---|---|---|---|
| `vllm_bf16` | 0.5~0.7 / 0.6~0.8 / 0.65~0.85 / 0.65~0.85 | 0.02~0.05 / 0.1~0.2 / 0.3~0.45 / 0.35~0.5 | 1~3 |
| `vllm_int4` | 0.35~0.55 / 0.45~0.65 / 0.5~0.7 / 0.5~0.7 | vllm_bf16과 같음 | 1~3 |
| `llamacpp_q4` | 0.3~0.5 / 0.35~0.6 / 0.4~0.65 / 0.4~0.65 | 0.01~0.03 / 0.08~0.15 / 0.2~0.35 / 0.25~0.4 | 2~5 |
| `hf_bf16` | 0.4~0.6 / 0.5~0.7 / 0.5~0.7 / 0.5~0.7 | 0.02~0.05 / 0.1~0.2 / 0.25~0.4 / 0.3~0.45 | 3~8 |

mid는 구간 중앙값입니다. 양자화 곡선을 쓰면 `warnings`에 "quant는 저장 크기이며 커널 속도는 보정 곡선이 반영한다"를 넣습니다.

### 6.4 saturating_batch

`Workload.concurrency`를 오름차순으로 보며 `bound`(mid)가 `memory`에서 `compute`로 처음 바뀌는 N입니다. 목록 안에서 바뀌지 않으면 None. 작은 모델에서 `overhead`가 지배하는 구간은 전환으로 세지 않습니다.

## 7. 배치 모드 (`modes.py`)

### 7.1 static (HF `generate`. 요청 N개가 같은 I, O로 함께 시작하고 함께 끝남)

```
prefill:  N × I 토큰을 한 번에.  T = N × I
          bytes = W_stream + KV_tok × N × I (쓰기)
          flops = 2 × P_body × N × I + 2 × V × h × N + 2 × L × q × N × I × (I + 1)
TTFT    = t_prefill
decode:   step j (0 ≤ j < D), c_j = I + 1 + j, Σc = N × c_j,  T = N
ITL     = mean_j step_ms(j)
resident = N,  M = 0
```

1차는 요청 길이가 모두 같다고 가정합니다. `O = 1`은 4.1에서 거부되므로 D ≥ 1입니다.

### 7.2 continuous (정상 상태. 평균 N개가 decode 중)

요청은 prefill 뒤 D step 동안 decode하고 떠나므로 step당 `λ = N / D`개가 새로 들어오고, 보존 법칙에 따라 step당 평균 `λ × I` prefill 토큰을 처리해야 합니다. decode 중 평균 컨텍스트는 `c̄ = I + 1 + (D − 1) / 2`입니다.

**mixes_prefill = True (vLLM chunked prefill. 토큰 예산 B에 decode 포함)**

스케줄링 정책: 매 step decode N개를 먼저 넣고, 남은 예산 `c = B − N`을 FCFS로 가장 오래된 prefill 요청에 주며, 그 요청이 끝나고 예산이 남으면 다음 요청에 넘깁니다. 예산은 버려지지 않으므로 보존 조건은 하나입니다.

```
c = B − N                                    # B = None이면 c = I (통째 prefill, 예산 제한 없음)
feasible_budget : c ≥ 1  and  N + λ × I ≤ B   # 위반하면 infeasible_reason = "token_budget"
k      = ceil(I / c)                          # 예산을 혼자 쓸 때 요청 하나의 prefill step 수 (TTFT 기준)
f      = λ × I / c                            # prefill 토큰이 들어 있는 step의 비율, 0 ≤ f ≤ 1
M      = ceil(λ × k)                          # prefill 중인 요청 수 상한 (Little)
R      = N + M

step 종류 두 가지:
  s_d          = decode만.   T = N,     bytes = W_stream + KV_tok × (N × c̄ + N) + (kv_realloc ? 2 × KV_tok × N × c̄ : 0)
                                        flops = 2 × P_mm × N + 4 × L × q × N × c̄
  s_mix(C, c₀) = decode + prefill 조각.  T = N + C
                                        bytes = s_d.bytes + KV_tok × (c₀ + C)                 # 조각의 이전 KV 읽기 + 새 KV 쓰기
                                        flops = s_d.flops + 2 × P_body × C + 4 × L × q × (C × c₀ + C × (C + 1) / 2) + (마지막 조각이면 2 × V × h)
  각각 step_ms = max(t_mem, t_comp) + overhead

TTFT  = Σ_{j=0}^{k−1} s_mix(C_j, j × c, last_j),  C_j = min(c, I − j × c)   # 요청이 예산을 혼자 쓸 때. 대기열 제외
ITL   = f × (TTFT / k) + (1 − f) × s_d                                        # 빈도 가중 평균. prefill이 섞인 step의 대표값은 실제 조각 step들의 평균
```

`c ≤ 0`(N ≥ B)이면 prefill을 표현할 수 없으므로 `ttft_ms = None`, `itl_ms = s_d`, `feasible = False`.

**mixes_prefill = False (prefill이 별도 pass로 decode를 멈춤. llama.cpp server 기본 가정)**

```
c         = token_budget (None이면 I)
t_p       = Σ_{j=0}^{k−1} step_ms(조각 j)     # 조각 j: C_j = min(c, I − j × c), c₀ = j × c, T = C_j, 가중치는 조각마다 읽되 lm_head는 마지막만
s_d       = decode-only step_ms (위와 같음)
ITL       = s_d + λ × t_p                     # step당 평균 λ개의 prefill pass가 끼어듦
TTFT      = t_p + 0.5 × s_d                   # 도착이 decode step 안에서 균등하다고 가정
M         = ceil(λ × t_p / ITL)
R         = N + M
feasible_budget = True                        # prefill pass는 시간을 쓰고 예산을 쓰지 않음
```

**용량 판정 (두 경우 공통)**

```
kv_needed = N × seq_kv_peak + M × KV_tok × I'            # I' = kv_prealloc_ctx 가 있으면 그 값, 없으면 roundup(I, kv_block)
feasible  = (kv_needed ≤ usable)  and  (max_seqs is None or R ≤ max_seqs)  and  feasible_budget
infeasible_reason = 처음 실패한 조건 순서대로 "memory" → "max_seqs" → "token_budget"
total_tok_s = N × 1000 / ITL                               # decode 출력만. prefill이 내는 첫 토큰(step당 λ개)은 제외
```

`assumptions`: "steady state; admission queueing excluded; equal request lengths; prefix caching off; FCFS prefill with leftover budget passed to the next request; total_tok_s counts decode tokens only". `feasible=False`인 점도 계산 가능한 숫자는 채워 "상한을 올리면 얼마나 나오는가"를 보여 줍니다.

## 8. 엔진 프로필 (`profiles.py`)

| 필드 | `vllm()` | `ollama()` | `hf()` |
|---|---|---|---|
| `version` | `"0.6.3"` | `"0.5.x (llama.cpp server)"` | `"transformers 4.46.x, dtype=bf16 명시 로드"` |
| `batching` | continuous | continuous | static |
| `max_seqs` | 256 (`max_num_seqs`) | 4 (`num_parallel` 인자. ollama 자동값은 4를 시도하고 VRAM 부족 시 1) | None |
| `quant` | None. config `quantization_config.quant_method`가 `awq`/`gptq`면 그 값 | `"gguf_q4_k_m"` (`quant` 인자로 변경) | None |
| `kv_bits` | 16 | 16 | 16 |
| `kv_block` | 16 | 0 | 0 |
| `kv_prealloc_ctx` | None | 2048 (`num_ctx` 인자) | None |
| `memory_budget_ratio` | 0.90 (`gpu_memory_utilization`) | 1.0 | 1.0 |
| `workspace_ratio` | 0.05 (추정) | 0.15 (추정) | 0.20 (추정) |
| `token_budget` | 512 (`chunked=True` 기본. 0.6.3은 `max_model_len > 32k`인 모델에서 자동 활성, 예산 512에 decode 포함. `chunked=False`면 None) | 512 (`num_batch`) | None |
| `mixes_prefill` | True (`chunked=False`면 False) | False (보수적 가정. 0.2 실측 항목) | False |
| `kv_realloc` | False | False | True |
| `calibration` | `vllm_bf16`, quant가 있으면 `vllm_int4` | `llamacpp_q4` | `hf_bf16` |

팩토리: `vllm(chunked=True, **overrides)`, `ollama(num_parallel=4, num_ctx=2048, quant="gguf_q4_k_m", **overrides)`, `hf(**overrides)`. 모두 frozen dataclass를 반환하고 `dataclasses.replace`로 바꿀 수 있습니다. `profile="ollama"`처럼 문자열이면 기본 팩토리를 호출합니다.

`quant.py` 테이블: `None → 16`, `fp8 → 8`, `awq → 4`, `gptq → config bits (기본 4)`, `gguf_q8_0 → 8.5`, `gguf_q6_k → 6.6`, `gguf_q5_k_m → 5.7`, `gguf_q4_k_m → 4.9`, `gguf_q4_0 → 4.5`. 각 항목에 보정 곡선 이름이 붙습니다. 목록 밖 문자열은 `ValueError`.

## 9. CLI (`cli.py`)

```
servcalc MODEL --gpu NAME [--engine vllm|ollama|hf] [--prompt 1024] [--gen 512]
         [--concurrency 1,4,16,64] [--quant awq] [--kv-bits 8] [--max-seqs 32]
         [--num-ctx 8192] [--token-budget 2048] [--no-chunked] [--slo-itl 50] [--json]
servcalc gpus
```

기본 출력은 요약(상주 가중치 GB, 스트리밍 GB, KV/토큰, memory capacity, configured capacity, saturating batch, `--slo-itl` 지정 시 최대 사용자 수)과 배치별 표(N, R, feasible 또는 사유, TTFT mid[low~high], ITL mid[low~high], seq tok/s, total tok/s, bound)입니다. `DoesNotFitError`는 exit 1, 인자 오류는 exit 2.

## 10. 검증과 에러 처리

| 상황 | 동작 |
|---|---|
| `usable ≤ 0` (가중치가 예산 초과) | `DoesNotFitError` (ValueError 하위), exit 1 |
| `gpu`가 숫자 또는 모르는 이름 | `ValueError` / `KeyError`(vramcalc 후보 안내) |
| 모르는 프로필 이름, 모르는 `quant` | `KeyError` / `ValueError`, 내장 목록 안내 |
| `prompt_len`, concurrency 값, `token_budget`, `max_seqs`가 양의 정수가 아님, `kv_block`이 음수, `gen_len < 2` | `ValueError` |
| `kv_bits`가 8, 16 밖 | `ValueError` |
| `memory_budget_ratio`, `workspace_ratio`가 (0, 1] 밖이거나 `workspace_ratio ≥ memory_budget_ratio` | `ValueError` |
| `GpuSpec` 필드가 양수가 아님, Curve x가 오름차순이 아니거나 low ≤ mid ≤ high 위반 | `ValueError` |
| `kv_prealloc_ctx < I + O` | `ValueError` (잘림 미지원) |
| `token_budget ≤ 1` (decode 1개 + prefill 최소 1토큰 불가) | `ValueError` |
| 지원 밖 model_type | vramcalc `UnsupportedModelError` 전파 |
| 양자화 곡선 사용, 가중치 바이트 근사 사용 | `warnings` |

## 11. 테스트

tiny Arch는 vramcalc와 같습니다(h=64, L=2, a=4, kv_heads=2, head_dim=16, i=128, V=256, untied).

- `test_work.py`: `W_res`/`W_stream`의 tie 분기와 양자화 비트, `KV_tok`, decode bytes의 가중치 부분이 N과 무관하고 KV 부분이 Σc에 비례, `kv_realloc` 가산, prefill `flops_total`이 조각 크기와 무관(k=1과 k=4가 같음), `kv_read_total` 공식, 마지막 조각에만 lm_head
- `test_timing.py`: Curve 보간과 clamp, 오름차순 검증, `max`와 `bound`, Range 방향(시간은 효율 high → low, 처리량은 역수)
- `test_modes.py`: static ITL이 j에 대해 단조 증가, 평균 컨텍스트 `c̄`, continuous의 `λ = N/D`, `feasible_budget` 경계(`N + λI = B`에서 True, +1에서 False), `c ≤ 0`일 때 `ttft_ms=None`, `f`와 빈도 가중 ITL(λ→0에서 ITL→s_d), TTFT가 조각별 `s_mix` 합, `B=None`과 `max_seqs=None` 처리, `kv_realloc` 항, `mixes_prefill=False`의 `t_p` 합과 `ITL = s_d + λ t_p`
- `test_capacity.py`: `usable` 식, `kv_prealloc_ctx`, `kv_block` 올림, `infeasible_reason` 순서, `DoesNotFitError`, 컨텍스트 초과 `ValueError`
- `test_profiles.py`: 세 프로필 기본값, 팩토리 인자, `replace`, `quantization_config` → `quant`, 우선순위(명시 > config > 기본)
- `test_estimate.py`: 실제 config 8개로 sanity 경계. Llama-3.1-8B + RTX 3090 + vLLM bf16: `memory_capacity_seqs ≥ 1`, N=1 ITL mid 15~40ms, `saturating_batch`는 None 또는 전환점 성질(그 N에서 compute, 직전 N에서 memory). `max_users`. `to_dict` 직렬화. 모든 Range가 `low ≤ mid ≤ high`
- `test_cli.py`: subprocess로 표, `--json`, `--slo-itl`, 안 들어가는 모델 exit 1, 인자 오류 exit 2

## 12. CI와 배포

vramcalc와 동일: `uv sync --extra dev`, `ruff check`, `pytest`를 Python 3.10/3.12에서. 태그 push 시 PyPI trusted publishing. 저장소 `sacom123/servcalc`, MIT, 영어 + 한국어 README. README에 "실측 전 사전 범위이며 정확도 수치를 주장하지 않는다"를 명시합니다.

## 13. 0.2 계획

- 실측: RunPod GPU 한 장에서 vLLM `benchmark_serving.py`와 llama.cpp `llama-bench`, `llama-server`로 세 실험. (1) 7~8B dense 모델 decode sweep N=1, 4, 16, 64, 짧은/긴 컨텍스트, bf16으로 엔진 간 동일 조건 후 양자화 추가. (2) prefill 길이 sweep과 토큰 예산, 두 I/O 비율의 포화 서빙. (3) 용량 경계 확인과 다른 모델 또는 GPU로 holdout. llama.cpp server가 prefill과 decode를 한 배치에 섞는지 확인해 `ollama` 프로필의 `mixes_prefill`을 확정. 결과로 보정 곡선을 `measured`로 교체하고 README에 표 게시
- 요청 도착률 시뮬레이터(대기열, 백분위), speculative decoding, prefix caching, TP, MoE(vramcalc 0.2의 `Arch` 확장 후), 양자화 포맷별 정확한 잔류 바이트, 추가 프로필

## 14. 구현 순서

1. 골격, `gpus.py`, `Workload`, `quant.py`, `Curve`/`Calibration` 보간, 테스트
2. `work.py` 작업량 식, 테스트
3. `timing.py` 시간 변환, bound, Range, 테스트
4. `modes.py` static, continuous, 용량과 feasibility, 테스트
5. `profiles.py` 세 프로필, 테스트
6. `estimate.py` 조합, 결과 객체, sanity 테스트
7. `cli.py`, README, CI, PyPI 배포
