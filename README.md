# servcalc

[![PyPI](https://img.shields.io/pypi/v/servcalc)](https://pypi.org/project/servcalc/) [![CI](https://github.com/sacom123/servcalc/actions/workflows/ci.yml/badge.svg)](https://github.com/sacom123/servcalc/actions/workflows/ci.yml) [![Python](https://img.shields.io/pypi/pyversions/servcalc)](https://pypi.org/project/servcalc/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](#english) | [한국어](#한국어)

---

## English

Predict how many concurrent users one GPU can serve for an LLM, and at what speed, before you rent the GPU. servcalc reads the Hugging Face `config.json`, a GPU name and a serving engine (vLLM, ollama, or plain `transformers`), and returns per batch size: resident sequences, time to first token (TTFT), inter-token latency (ITL), tokens per second, and whether that batch size is feasible on the chosen engine. It is the next question after [vramcalc](https://github.com/sacom123/vramcalc): vramcalc answers "does it fit", servcalc answers "how fast, for how many".

```python
from servcalc import Workload, estimate

e = estimate("meta-llama/Llama-3.1-8B", "RTX 3090", workload=Workload(prompt_len=1024, gen_len=512))
print(e.table())
print(e.max_users(itl_ms=50))          # largest feasible batch whose mean ITL is within 50 ms
for p in e.points:
    print(p.batch, p.feasible, p.itl_ms.mid, p.total_tok_s.mid)
```

```
$ servcalc meta-llama/Llama-3.1-8B --gpu "RTX 3090" --concurrency 1,4,16,64 --slo-itl 50
Llama-3.1-8B on RTX 3090 via vllm (0.6.3)
weights 16.06 GB resident, 15.01 GB streamed per step; KV 131.1 KB/token
capacity: memory 29 seqs, configured 29 seqs; saturating batch 16
calibration: prior (ranges are low~high)

    N     R ok                              TTFT ms                 ITL ms            seq tok/s            total tok/s bound
    1     2 yes                       579 [482~724]       29.7 [24.7~36.2]     33.7 [27.6~40.5]             34 [28~40] memory
    4     5 yes                       600 [493~768]       30.8 [25.8~43.5]     32.5 [23.0~38.8]           130 [92~155] memory
   16    17 yes                       638 [522~821]      68.6 [50.8~105.2]      14.6 [9.5~19.7]          233 [152~315] compute
   64    65 no:memory                 747 [612~956]    139.9 [109.6~194.3]        7.1 [5.1~9.1]          457 [329~584] compute

max users with mean ITL <= 50 ms: 4
```

`N` is the number of sequences in decode, `R` the sequences resident in memory including the ones being prefilled. `ok` names the first limit hit: `memory`, `max_seqs`, or `token_budget`. Every number is a `[low~high]` range around a mid value.

### Install

| Command | What you get |
|---|---|
| `pip install servcalc` | Estimator and CLI. Depends only on `vramcalc`. Takes a local `config.json` or a model directory |
| `pip install "servcalc[hf]"` | Hugging Face Hub repo ids as input (`huggingface_hub`) |

Offline or on-premises: pass a model directory. With `HF_HUB_OFFLINE=1`, Hub ids resolve from the local cache only.

### Why

Serving questions arrive in the same breath as memory questions: "Will an 8B model on a 3090 handle twenty users at a readable speed?" The usual answer is to rent the GPU, start vLLM and run a load test. That works, but it costs an afternoon per configuration, and the result says nothing about why the number is what it is. servcalc writes the roofline model down explicitly. Decode steps are bounded by memory bandwidth (weights and KV cache read once per step), prefill is bounded by compute, and an engine's scheduler decides how the two share a step. Each term is a formula you can read in `docs/design.md`, so when a measurement disagrees you know which assumption to fix.

### Scope

- Models: dense decoder-only models that vramcalc parses (`llama`, `mistral`, `qwen2`, `gemma`, `gemma2`). GQA, tied embeddings and QKV bias are read from the config.
- GPUs: 34 cards from RTX 3060 to B200, with memory bandwidth and dense bf16 TFLOPs from vendor spec sheets. Pass a `GpuSpec` for anything else.
- Engines, as value-only profiles: `vllm` 0.6.3 (continuous batching, chunked prefill with a 512-token budget, paged KV in blocks of 16, `max_num_seqs` 256), `ollama` 0.5.x (llama.cpp server, GGUF Q4_K_M by default, `num_parallel` 4, `num_ctx` 2048 preallocated per slot), `hf` (`transformers` `generate()` in bf16, static batching, KV cache reallocated every step).
- Workload: equal prompt and generation lengths for every request, a list of batch sizes to sweep.

Out of scope for 0.1: MoE, MLA and sliding-window attention, speculative decoding, prefix caching, multi-GPU (tensor or pipeline parallel), CPU offload, admission queueing, and request length distributions. These are all deliberate: each one is a separate term or a separate model, and 0.1 keeps the closed-form core small enough to validate.

### How it is computed

**Work.** For a decode step with `N` sequences and total context `sum_ctx`, bytes moved are `W_stream + KV_tok x sum_ctx + KV_tok x N`, where `W_stream` is the weight bytes read per step (input embedding rows are a lookup and excluded) and `KV_tok` the KV bytes per token. FLOPs are `2 x P_mm x N + 4 x L x q x sum_ctx`. A prefill chunk of `C` tokens with `c0` tokens already cached costs `2 x P_body x C + 4 x L x q x (C x c0 + C(C+1)/2)` FLOPs and writes `KV_tok x C` bytes. Summing chunks reproduces the whole-prompt cost exactly, whatever the chunk size.

**Time.** `step_ms = max(bytes / (bandwidth x bw_util), flops / (tflops x mfu)) + overhead`. `bw_util` and `mfu` are efficiency curves over the number of tokens in the step; `overhead` is the per-step scheduler and launch cost. Each curve has a low, mid and high value, which is where the ranges come from. The step is labelled `memory`, `compute` or `overhead` by its largest term at mid, and `saturating batch` is the first batch where decode flips from memory- to compute-bound.

**Modes.** Static batching (hf) runs one prefill for the whole batch, then `gen_len - 1` decode steps whose context grows each step; ITL is their mean. Continuous batching (vllm, ollama) is solved in steady state: with `N` sequences each producing `gen_len - 1` tokens, requests arrive at `N / (gen_len - 1)` per step. When the engine mixes prefill into decode steps (vLLM chunked prefill), the prompt is split into `k = ceil(prompt_len / (budget - N))` chunks and TTFT is the sum of those mixed steps; ITL is the frequency-weighted mean of mixed and decode-only steps. When prefill runs as its own pass (llama.cpp), ITL is `s_decode + arrival_rate x t_prefill` and TTFT is `t_prefill + s_decode / 2`.

**Profiles and capacity.** A profile is a dataclass of values: batching mode, sequence cap, quantization, KV bits and block size, memory budget and workspace ratio, token budget, whether prefill is mixed, whether KV is reallocated. Capacity is `(VRAM x budget - weights - VRAM x workspace) / KV per sequence at peak length`, rounded to KV blocks or to `num_ctx` for ollama. A batch is infeasible when its resident sequences exceed memory, the sequence cap, or the token budget, in that order.

### Accuracy

**All numbers in 0.1 are prior ranges.** The efficiency curves are informed guesses from published vLLM and llama.cpp benchmarks, not measurements, and `calibration: prior` in the output says so. servcalc does not claim an error figure yet. 0.2 will run vLLM's `benchmark_serving.py` and llama.cpp's server benchmark on rented GPUs, fit the curves per engine and quantization, and report the residual the same way vramcalc reports its 19-run table. Until then, read the ranges as "the model is somewhere in here if the formulas hold", and use the `bound` column and `saturating batch` as the qualitative result.

### Limits

- The steady-state solution is a fluid approximation: it assumes arrivals spread evenly across steps. Real schedulers admit requests in bursts, which widens TTFT.
- Quantized weight bytes count non-linear tensors (embeddings, norms) as 16-bit. Pass `weight_bytes=` with the real checkpoint size if you have it.
- `bf16_tflops` for T4 and V100 is the fp16 tensor-core peak since those cards have no bf16.
- Whether ollama's llama.cpp build mixes prefill into decode steps depends on its version and flags. 0.1 models it as a separate pass; 0.2 will confirm.

### License

MIT

---

## 한국어

GPU를 빌리기 전에, 하나의 GPU가 LLM 사용자를 동시에 몇 명까지 어느 속도로 서빙할 수 있는지 예측하는 라이브러리입니다. Hugging Face `config.json`, GPU 이름, 서빙 엔진(vLLM, ollama, 일반 `transformers`)을 입력받아 배치 크기별로 상주 시퀀스 수, 첫 토큰 시간(TTFT), 토큰 간 지연(ITL), 초당 토큰 수, 그리고 해당 배치가 선택한 엔진에서 실행 가능한지를 반환합니다. [vramcalc](https://github.com/sacom123/vramcalc)의 다음 질문에 해당합니다. vramcalc는 "들어가는가"에, servcalc는 "얼마나 빠르게, 몇 명에게"에 답합니다.

```python
from servcalc import Workload, estimate

e = estimate("meta-llama/Llama-3.1-8B", "RTX 3090", workload=Workload(prompt_len=1024, gen_len=512))
print(e.table())
print(e.max_users(itl_ms=50))          # 평균 ITL이 50 ms 이내인 가장 큰 실행 가능 배치
for p in e.points:
    print(p.batch, p.feasible, p.itl_ms.mid, p.total_tok_s.mid)
```

```
$ servcalc meta-llama/Llama-3.1-8B --gpu "RTX 3090" --concurrency 1,4,16,64 --slo-itl 50
Llama-3.1-8B on RTX 3090 via vllm (0.6.3)
weights 16.06 GB resident, 15.01 GB streamed per step; KV 131.1 KB/token
capacity: memory 29 seqs, configured 29 seqs; saturating batch 16
calibration: prior (ranges are low~high)

    N     R ok                              TTFT ms                 ITL ms            seq tok/s            total tok/s bound
    1     2 yes                       579 [482~724]       29.7 [24.7~36.2]     33.7 [27.6~40.5]             34 [28~40] memory
    4     5 yes                       600 [493~768]       30.8 [25.8~43.5]     32.5 [23.0~38.8]           130 [92~155] memory
   16    17 yes                       638 [522~821]      68.6 [50.8~105.2]      14.6 [9.5~19.7]          233 [152~315] compute
   64    65 no:memory                 747 [612~956]    139.9 [109.6~194.3]        7.1 [5.1~9.1]          457 [329~584] compute

max users with mean ITL <= 50 ms: 4
```

`N`은 decode 중인 시퀀스 수, `R`은 prefill 중인 것까지 포함해 메모리에 상주하는 시퀀스 수입니다. `ok`는 처음 걸리는 한계를 표시합니다(`memory`, `max_seqs`, `token_budget`). 모든 수치는 중앙값과 `[low~high]` 범위로 제공됩니다.

### 설치

| 명령 | 제공 내용 |
|---|---|
| `pip install servcalc` | 추정기와 CLI. 의존성은 `vramcalc` 하나입니다. 로컬 `config.json` 또는 모델 디렉터리를 입력받습니다 |
| `pip install "servcalc[hf]"` | Hugging Face Hub 저장소 id 입력(`huggingface_hub`) |

오프라인 또는 온프레미스 환경에서는 모델 디렉터리를 전달하면 됩니다. `HF_HUB_OFFLINE=1`을 설정하면 Hub id는 로컬 캐시에서만 해석됩니다.

### 만든 이유

서빙에 관한 질문은 메모리 질문과 함께 옵니다. "3090에서 8B 모델로 사용자 20명을 읽을 만한 속도로 받을 수 있는가"가 대표적입니다. 통상적인 답은 GPU를 빌려 vLLM을 올리고 부하 시험을 돌리는 것입니다. 유효한 방법이지만 구성 하나에 반나절이 들고, 결과 수치가 왜 그 값인지는 알려 주지 않습니다. servcalc는 roofline 모델을 명시적으로 기술합니다. decode step은 메모리 대역폭(매 step 가중치와 KV 캐시를 한 번 읽음)에, prefill은 연산량에 묶이며, 엔진의 스케줄러가 둘을 한 step에서 어떻게 섞는지를 결정합니다. 각 항은 `docs/design.md`에 식으로 적혀 있으므로, 실측과 다를 때 어느 가정을 고쳐야 하는지 알 수 있습니다.

### 지원 범위

- 모델: vramcalc가 해석하는 dense decoder-only 모델(`llama`, `mistral`, `qwen2`, `gemma`, `gemma2`). GQA, tied embedding, QKV bias는 config에서 읽습니다.
- GPU: RTX 3060부터 B200까지 34종. 메모리 대역폭과 dense bf16 TFLOPs는 제조사 사양표 기준입니다. 그 외 장비는 `GpuSpec`으로 전달합니다.
- 엔진(값으로만 구성된 프로필): `vllm` 0.6.3(continuous batching, 512 토큰 예산의 chunked prefill, 16 토큰 블록 단위 paged KV, `max_num_seqs` 256), `ollama` 0.5.x(llama.cpp server, 기본 GGUF Q4_K_M, `num_parallel` 4, 슬롯당 `num_ctx` 2048 선할당), `hf`(`transformers` `generate()` bf16, static batching, 매 step KV 캐시 재할당).
- 워크로드: 모든 요청의 prompt 길이와 생성 길이가 같다고 가정하며, 배치 크기 목록을 순회합니다.

0.1에서 제외한 항목: MoE, MLA, sliding-window attention, speculative decoding, prefix caching, 다중 GPU(tensor 또는 pipeline parallel), CPU offload, 대기열, 요청 길이 분포. 각각 별도의 항 또는 별도의 모델이 필요하므로, 0.1은 검증 가능한 크기의 닫힌 식 핵심만 담았습니다.

### 계산 구조

**작업량.** 시퀀스 `N`개, 총 컨텍스트 `sum_ctx`인 decode step의 이동 바이트는 `W_stream + KV_tok x sum_ctx + KV_tok x N`입니다. `W_stream`은 step마다 읽는 가중치 바이트(입력 embedding은 조회이므로 제외), `KV_tok`은 토큰당 KV 바이트입니다. FLOPs는 `2 x P_mm x N + 4 x L x q x sum_ctx`입니다. 이미 `c0` 토큰이 캐시된 상태에서 `C` 토큰을 prefill하는 조각은 `2 x P_body x C + 4 x L x q x (C x c0 + C(C+1)/2)` FLOPs를 소비하고 `KV_tok x C` 바이트를 기록합니다. 조각의 합은 조각 크기와 무관하게 전체 prompt 비용과 정확히 일치합니다.

**시간.** `step_ms = max(bytes / (bandwidth x bw_util), flops / (tflops x mfu)) + overhead`입니다. `bw_util`과 `mfu`는 step 내 토큰 수에 대한 효율 곡선이고, `overhead`는 step당 스케줄러와 커널 실행 비용입니다. 각 곡선이 low, mid, high 값을 가지므로 결과가 범위로 나옵니다. step은 mid 기준으로 가장 큰 항에 따라 `memory`, `compute`, `overhead`로 표시되고, `saturating batch`는 decode가 memory-bound에서 compute-bound로 바뀌는 첫 배치입니다.

**모드.** static batching(hf)은 배치 전체를 한 번 prefill한 뒤 컨텍스트가 매 step 늘어나는 `gen_len - 1`회의 decode step을 수행하며, ITL은 그 평균입니다. continuous batching(vllm, ollama)은 정상 상태로 풉니다. 시퀀스 `N`개가 각각 `gen_len - 1` 토큰을 생성하므로 요청은 step당 `N / (gen_len - 1)`개 도착합니다. 엔진이 prefill을 decode step에 섞는 경우(vLLM chunked prefill) prompt는 `k = ceil(prompt_len / (budget - N))`개 조각으로 나뉘고 TTFT는 그 혼합 step들의 합, ITL은 혼합 step과 decode 전용 step의 빈도 가중 평균입니다. prefill이 별도 pass로 실행되는 경우(llama.cpp) ITL은 `s_decode + arrival_rate x t_prefill`, TTFT는 `t_prefill + s_decode / 2`입니다.

**프로필과 용량.** 프로필은 값만 담은 dataclass입니다. batching 방식, 시퀀스 상한, 양자화, KV 비트와 블록 크기, 메모리 예산과 workspace 비율, 토큰 예산, prefill 혼합 여부, KV 재할당 여부가 들어갑니다. 용량은 `(VRAM x budget - weights - VRAM x workspace) / 최대 길이 시 시퀀스당 KV`이며, KV 블록 또는 ollama의 `num_ctx` 단위로 올림합니다. 상주 시퀀스가 메모리, 시퀀스 상한, 토큰 예산을 이 순서로 초과하면 해당 배치는 실행 불가로 표시됩니다.

### 정확도

**0.1의 모든 수치는 실측 전 사전 범위입니다.** 효율 곡선은 공개된 vLLM과 llama.cpp 벤치마크를 참고한 추정값이며 측정값이 아닙니다. 출력의 `calibration: prior`가 이를 표시합니다. servcalc는 아직 오차 수치를 주장하지 않습니다. 0.2에서는 임대 GPU에서 vLLM의 `benchmark_serving.py`와 llama.cpp server 벤치마크를 실행해 엔진과 양자화별로 곡선을 맞추고, vramcalc의 19회 실측 표와 같은 방식으로 잔차를 보고할 예정입니다. 그때까지 범위는 "식이 맞다면 이 안에 있다"로 읽고, `bound` 열과 `saturating batch`를 정성적 결과로 활용하시기 바랍니다.

### 한계

- 정상 상태 해는 유체 근사입니다. 요청 도착이 step에 균등하게 퍼진다고 가정하므로, 실제 스케줄러의 묶음 단위 admission은 TTFT를 더 넓게 만듭니다.
- 양자화 가중치 바이트는 비선형 텐서(embedding, norm)를 16비트로 계산합니다. 실제 체크포인트 크기를 알고 있다면 `weight_bytes=`로 전달하시기 바랍니다.
- T4와 V100은 bf16이 없으므로 `bf16_tflops`에 fp16 tensor core 피크를 사용합니다.
- ollama의 llama.cpp 빌드가 prefill을 decode step에 섞는지는 버전과 옵션에 따라 다릅니다. 0.1은 별도 pass로 모델링하며 0.2에서 확인합니다.

### 라이선스

MIT
