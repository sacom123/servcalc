# servcalc 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** config.json과 GPU 사양으로 단일 GPU LLM 서빙 용량(동시 시퀀스 수, tok/s, TTFT, ITL)을 배치 크기별 범위로 예측하는 `servcalc` 0.1.0을 완성합니다.

**Architecture:** vramcalc의 `Arch`, `load_arch`, `kv_cache`, GPU 이름 해석을 재사용합니다. `work.py`의 순수 함수가 바이트와 FLOPs를 내고, `timing.py`가 효율 곡선으로 시간 Range로 바꾸며, `modes.py`가 static과 continuous 정상 상태 식과 용량 판정을 수행합니다. `profiles.py`가 엔진 차이를 값으로 담고, `estimate.py`가 조합합니다.

**Tech Stack:** Python 3.10+, uv, hatchling, pytest, ruff. 의존성 `vramcalc>=0.1.2`.

**Spec:** `docs/design.md`

## Global Constraints

- Python `>=3.10`. 기본 의존성 `vramcalc>=0.1.2` 하나. extras: `hf = ["vramcalc[hf]"]`, `dev = ["pytest", "ruff"]`.
- 내부 단위: 바이트와 FLOPs는 float, 시간은 ms, 처리량은 tok/s. 출력 시 GB(10^9)로 변환.
- `gen_len ≥ 2`, `D = O − 1`, 평균 decode 컨텍스트 `c̄ = I + 1 + (D − 1) / 2`.
- 모든 `Range`는 `low ≤ mid ≤ high`. 시간은 효율 high → low, 처리량은 역수.
- 지원 밖 설정은 거부: `DoesNotFitError`(ValueError 하위), `ValueError`, `KeyError`. 조용한 근사는 `warnings`에만.
- 커밋은 Conventional Commits, AI attribution 없음. 작성자 `Seonho Hong <16527470+sacom123@users.noreply.github.com>`(저장소 로컬 설정 완료).
- 코드 주석과 docstring은 영어, 문서는 한국어 업무체. 도구 마커 없음.

## Review Focus

1. `token_budget=None` + `mixes_prefill=True`(vLLM `chunked=False`): `c = I`, `k = 1`, `f = λI/I = λ`. λ > 1(N > D)이면 `f > 1`이 되어 가중 평균이 깨짐. `f = min(1, λI/c)`로 clamp하고 `N + λI ≤ B` 검사는 B=None이면 생략. (Task 4에서 테스트)
2. `max_seqs=None`인 프로필에서 `configured_capacity_seqs`는 `memory_capacity_seqs`와 같아야 하며 `R ≤ max_seqs` 검사는 생략. (Task 4)
3. 사용자가 `--gpu 24`처럼 숫자를 넘기면 `ValueError`에 "GpuSpec 필요" 안내. 이름 `A6000`, `a6000`, `RTX A6000`은 모두 같은 항목. (Task 1, 7)
4. `quantization_config`에 `quant_method`가 없거나 모르는 값(`bitsandbytes` 등)이면 `ValueError`로 거부. (Task 5)
5. 양자화 프로필에서 `weight_bytes` override를 주면 `W_res`는 override, `W_stream`은 `override − (untied ? V·h·2 : 0)`을 음수가 되지 않게 clamp. (Task 2)

---

### Task 1: vramcalc 0.1.2 (A6000 계열 GPU, `load_config` 공개) 와 servcalc 골격

**Files:**
- Modify (vramcalc 저장소): `src/vramcalc/gpus.py`, `src/vramcalc/loaders.py`, `src/vramcalc/__init__.py`, `tests/test_gpus.py`, `tests/test_loaders.py`, `pyproject.toml`, `CHANGELOG.md`
- Create (servcalc): `pyproject.toml`, `README.md`(한 줄), `src/servcalc/__init__.py`, `src/servcalc/gpus.py`, `src/servcalc/workload.py`, `src/servcalc/quant.py`, `src/servcalc/calibration.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/configs/*.json`(vramcalc 복사), `tests/test_gpus.py`, `tests/test_workload.py`, `tests/test_quant.py`, `tests/test_calibration.py`

**Interfaces:**
- Produces (vramcalc 0.1.2): `vramcalc.gpus.GPUS`에 `RTX A6000`(48), `RTX 6000 Ada`(48), `A800 80GB`(80), `H800 80GB`(80) 추가; `vramcalc.loaders.load_config(model) -> dict` 공개
- Produces (servcalc): `GpuSpec(name, vram_bytes, bandwidth_gbs, bf16_tflops)`, `gpus.SPECS: dict[str, tuple[float, float]]`, `gpus.resolve(gpu) -> GpuSpec`; `Workload`; `quant.QUANTS: dict[str | None, tuple[float, str]]`(비트, 보정 접미사), `quant.bits(name) -> float`, `quant.curve_suffix(name) -> str`; `Curve(points)`, `Curve.at(x) -> tuple[float, float, float]`, `Calibration(bw_util, mfu, overhead_ms, source)`, `calibration.PRIORS: dict[str, Calibration]`, `calibration.get(name) -> Calibration`

- [ ] **Step 1: vramcalc에 실패 테스트 추가**

`tests/test_gpus.py` 끝:
```python
@pytest.mark.parametrize("query", ["A6000", "a6000", "RTX A6000", "NVIDIA RTX A6000"])
def test_lookup_a6000(query):
    assert gpus.lookup(query) == ("RTX A6000", 48 * 2**30)
```
`tests/test_loaders.py` 끝:
```python
from vramcalc.loaders import load_config


def test_load_config_returns_raw_dict(tmp_path, tiny_cfg):
    tiny_cfg["quantization_config"] = {"quant_method": "awq", "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(tiny_cfg))
    cfg = load_config(tmp_path)
    assert cfg["quantization_config"]["quant_method"] == "awq"
    assert load_config(tiny_cfg) is tiny_cfg
```

- [ ] **Step 2: 실패 확인**

Run: `cd /Users/seonho/Documents/code/library/vramcalc && uv run pytest tests/test_gpus.py tests/test_loaders.py -q`
Expected: FAIL (`unknown GPU 'A6000'`, `ImportError: cannot import name 'load_config'`)

- [ ] **Step 3: vramcalc 구현**

`gpus.py`의 dict에 추가: `"RTX A6000": 48, "RTX 6000 Ada": 48, "A800 80GB": 80, "H800 80GB": 80`.

`loaders.py`: `load_arch`의 본문을 `load_config`로 분리합니다.
```python
def load_config(model) -> dict:
    """Resolve dict / local path / Hub id into the raw config.json dict."""
    if isinstance(model, dict):
        return model
    path = Path(model)
    if path.is_dir():
        config = path / "config.json"
        if not config.is_file():
            raise FileNotFoundError(f"no config.json in directory {path}")
        return _read_json(config)
    if path.is_file():
        return _read_json(path)
    return _download_config(str(model))


def load_arch(model) -> Arch:
    if isinstance(model, Arch):
        return model
    cfg = load_config(model)
    arch = Arch.from_config(cfg)
    if arch.name or isinstance(model, dict):
        return arch
    path = Path(str(model))
    fallback = path.name if path.is_dir() else path.parent.name if path.is_file() else str(model).split("/")[-1]
    return _named(arch, fallback)
```
`__init__.py`에 `load_config` export. `pyproject.toml` 버전 `0.1.2`, `__version__` 동일, CHANGELOG에 `## 0.1.2 (2026-10-11)`: "Add RTX A6000, RTX 6000 Ada, A800, H800 to the GPU table; expose `load_config`".

- [ ] **Step 4: vramcalc 통과, 커밋, 태그, push**

Run: `uv run pytest -q && uv run ruff check .`
Expected: 전체 PASS
```bash
git add -A && git commit -m "feat: add A6000-class GPUs and expose load_config

Release 0.1.2." && git tag -a v0.1.2 -m "vramcalc 0.1.2" && git push origin main --tags
```
CI publish 완료 후 `pip index versions vramcalc`에 0.1.2가 보일 때까지 기다립니다.

- [ ] **Step 5: servcalc 골격**

`pyproject.toml`:
```toml
[project]
name = "servcalc"
version = "0.1.0"
description = "Predict single-GPU LLM serving capacity (concurrency, tok/s, TTFT, ITL) from config.json with a roofline model"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = ["vramcalc>=0.1.2"]
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.optional-dependencies]
hf = ["vramcalc[hf]"]
dev = ["pytest", "ruff"]

[project.scripts]
servcalc = "servcalc.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/servcalc"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```
```bash
cd /Users/seonho/Documents/code/library/servcalc
mkdir -p src/servcalc tests/configs
printf '"""servcalc: predict single-GPU LLM serving capacity from config.json."""\n\n__version__ = "0.1.0"\n' > src/servcalc/__init__.py
printf '# servcalc\n' > README.md
touch tests/__init__.py
cp ../vramcalc/tests/configs/*.json tests/configs/
uv sync --extra dev && uv run pytest -q
```
Expected: `no tests ran`

`tests/conftest.py`:
```python
import json
from pathlib import Path

import pytest

from vramcalc import Arch

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())


@pytest.fixture
def tiny_cfg() -> dict:
    return {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 128, "vocab_size": 256, "tie_word_embeddings": False,
    }


@pytest.fixture
def tiny(tiny_cfg) -> Arch:
    return Arch.from_config(tiny_cfg)
```

- [ ] **Step 6: servcalc 실패 테스트 작성**

`tests/test_gpus.py`:
```python
import pytest

from servcalc.gpus import GpuSpec, SPECS, resolve

from vramcalc import gpus as vg


def test_every_vramcalc_gpu_has_specs():
    missing = [n for n in vg.GPUS if n not in SPECS]
    assert missing == []


@pytest.mark.parametrize("query", ["RTX 3090", "3090", "rtx-3090"])
def test_resolve_name(query):
    g = resolve(query)
    assert g == GpuSpec("RTX 3090", 24 * 2**30, 936.0, 71.0)


def test_resolve_a6000():
    assert resolve("A6000").bandwidth_gbs == 768.0


def test_resolve_passthrough():
    g = GpuSpec("custom", 10**9, 100.0, 1.0)
    assert resolve(g) is g


def test_resolve_number_rejected():
    with pytest.raises(ValueError, match="GpuSpec"):
        resolve(24)


@pytest.mark.parametrize("field,value", [("vram_bytes", 0), ("bandwidth_gbs", -1.0), ("bf16_tflops", 0.0)])
def test_gpuspec_validation(field, value):
    kwargs = {"name": "x", "vram_bytes": 1, "bandwidth_gbs": 1.0, "bf16_tflops": 1.0, field: value}
    with pytest.raises(ValueError, match=field):
        GpuSpec(**kwargs)
```

`tests/test_workload.py`:
```python
import pytest

from servcalc.workload import Workload


def test_defaults():
    w = Workload(prompt_len=1024, gen_len=512)
    assert w.concurrency == (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert w.decode_steps == 511
    assert w.mean_decode_ctx == 1024 + 1 + 510 / 2


@pytest.mark.parametrize("kwargs,match", [
    ({"prompt_len": 0, "gen_len": 2}, "prompt_len"),
    ({"prompt_len": 8, "gen_len": 1}, "gen_len"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": (0, 1)}, "concurrency"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": (4, 2)}, "ascending"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": ()}, "concurrency"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Workload(**kwargs)
```

`tests/test_quant.py`:
```python
import pytest

from servcalc import quant


def test_table():
    assert quant.bits(None) == 16
    assert quant.bits("awq") == 4
    assert quant.bits("gguf_q4_k_m") == 4.9
    assert quant.curve_suffix(None) == "bf16"
    assert quant.curve_suffix("gguf_q4_k_m") == "q4"
    assert quant.curve_suffix("awq") == "int4"


def test_unknown():
    with pytest.raises(ValueError, match="bitsandbytes"):
        quant.bits("bitsandbytes")
```

`tests/test_calibration.py`:
```python
import pytest

from servcalc.calibration import PRIORS, Calibration, Curve, get


def test_curve_interpolates_and_clamps():
    c = Curve(((1, 0.5, 0.6, 0.7), (8, 0.6, 0.7, 0.8)))
    assert c.at(1) == (0.5, 0.6, 0.7)
    assert c.at(8) == (0.6, 0.7, 0.8)
    lo, mid, hi = c.at(4.5)
    assert abs(mid - 0.65) < 1e-9
    assert c.at(0) == (0.5, 0.6, 0.7)      # clamp below
    assert c.at(100) == (0.6, 0.7, 0.8)    # clamp above


@pytest.mark.parametrize("points,match", [
    (((8, 0.5, 0.6, 0.7), (1, 0.6, 0.7, 0.8)), "ascending"),
    (((1, 0.7, 0.6, 0.5),), "low <= mid <= high"),
    (((1, 0.0, 0.6, 0.7),), "positive"),
    ((), "at least one"),
])
def test_curve_validation(points, match):
    with pytest.raises(ValueError, match=match):
        Curve(points)


def test_priors_exist_and_are_prior():
    for name in ("vllm_bf16", "vllm_int4", "llamacpp_q4", "hf_bf16"):
        cal = get(name)
        assert isinstance(cal, Calibration)
        assert cal.source == "prior"
        lo, mid, hi = cal.overhead_ms
        assert lo <= mid <= hi
    with pytest.raises(KeyError, match="vllm_bf16"):
        get("nope")


def test_vllm_bf16_values():
    cal = PRIORS["vllm_bf16"]
    assert cal.bw_util.at(1) == (0.5, 0.6, 0.7)
    assert cal.mfu.at(512) == (0.3, 0.375, 0.45)
    assert cal.overhead_ms == (1.0, 2.0, 3.0)
```

- [ ] **Step 7: 실패 확인**

Run: `uv run pytest -q`
Expected: 4개 모듈 모두 `ModuleNotFoundError`

- [ ] **Step 8: 구현**

`src/servcalc/gpus.py`:
```python
"""GPU specs: vramcalc's capacity table extended with bandwidth and bf16 compute."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vramcalc import gpus as _vg

# (memory bandwidth GB/s, dense bf16 TFLOPs with fp32 accumulate). Vendor spec sheets, no sparsity.
SPECS: dict[str, tuple[float, float]] = {
    "RTX 3060": (360, 25), "RTX 3070": (448, 40), "RTX 3080": (760, 60), "RTX 3080 Ti": (912, 68),
    "RTX 3090": (936, 71), "RTX 3090 Ti": (1008, 80),
    "RTX 4060": (272, 30), "RTX 4060 Ti": (288, 44), "RTX 4070": (504, 58), "RTX 4070 Ti": (504, 80),
    "RTX 4070 Ti Super": (672, 88), "RTX 4080": (717, 97), "RTX 4080 Super": (736, 104), "RTX 4090": (1008, 165),
    "RTX 5070": (672, 62), "RTX 5070 Ti": (896, 88), "RTX 5080": (960, 112), "RTX 5090": (1792, 210),
    "T4": (320, 65), "V100 16GB": (900, 125), "V100 32GB": (900, 125),   # no bf16: fp16 tensor peak
    "A10": (600, 125), "A40": (696, 150), "RTX A6000": (768, 155), "RTX 6000 Ada": (960, 364),
    "A100 40GB": (1555, 312), "A100 80GB": (2039, 312), "A800 80GB": (2039, 312),
    "H100 80GB": (3350, 989), "H800 80GB": (3350, 989), "H200": (4800, 989),
    "L4": (300, 121), "L40S": (864, 362), "B200": (8000, 2250),
}


@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_bytes: int
    bandwidth_gbs: float
    bf16_tflops: float

    def __post_init__(self):
        for f in ("vram_bytes", "bandwidth_gbs", "bf16_tflops"):
            v = getattr(self, f)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{f} must be a positive finite number, got {v!r}")


def resolve(gpu) -> GpuSpec:
    if isinstance(gpu, GpuSpec):
        return gpu
    if not isinstance(gpu, str):
        raise ValueError(
            "gpu must be a name from the built-in table or a GpuSpec with bandwidth and TFLOPs; "
            f"a bare number ({gpu!r}) cannot be timed")
    name, vram = _vg.lookup(gpu)
    bw, tf = SPECS[name]
    return GpuSpec(name, vram, float(bw), float(tf))
```

`src/servcalc/workload.py`:
```python
"""Request shape: prompt and generation lengths plus the concurrency sweep."""

from __future__ import annotations

from dataclasses import dataclass


def _positive_int(name: str, v) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")


@dataclass(frozen=True)
class Workload:
    prompt_len: int
    gen_len: int
    concurrency: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    def __post_init__(self):
        _positive_int("prompt_len", self.prompt_len)
        if not isinstance(self.gen_len, int) or self.gen_len < 2:
            raise ValueError(f"gen_len must be an integer >= 2 (prefill emits the first token), got {self.gen_len!r}")
        if not self.concurrency:
            raise ValueError("concurrency must contain at least one value")
        for n in self.concurrency:
            _positive_int("concurrency", n)
        if list(self.concurrency) != sorted(set(self.concurrency)):
            raise ValueError("concurrency must be strictly ascending")

    @property
    def decode_steps(self) -> int:
        return self.gen_len - 1

    @property
    def mean_decode_ctx(self) -> float:
        """Average context during decode: step j has I + 1 + j for 0 <= j < D."""
        return self.prompt_len + 1 + (self.decode_steps - 1) / 2
```

`src/servcalc/quant.py`:
```python
"""Weight quantization formats: storage bits per linear weight and the calibration family."""

from __future__ import annotations

# name -> (bits per linear weight, calibration curve suffix)
QUANTS: dict[str | None, tuple[float, str]] = {
    None: (16, "bf16"),
    "fp8": (8, "bf16"),
    "awq": (4, "int4"),
    "gptq": (4, "int4"),
    "gguf_q8_0": (8.5, "q8"),
    "gguf_q6_k": (6.6, "q4"),
    "gguf_q5_k_m": (5.7, "q4"),
    "gguf_q4_k_m": (4.9, "q4"),
    "gguf_q4_0": (4.5, "q4"),
}


def _entry(name: str | None) -> tuple[float, str]:
    if name not in QUANTS:
        known = ", ".join(str(k) for k in QUANTS if k)
        raise ValueError(f"unknown quant {name!r}; supported: none, {known}")
    return QUANTS[name]


def bits(name: str | None) -> float:
    return _entry(name)[0]


def curve_suffix(name: str | None) -> str:
    return _entry(name)[1]
```

`src/servcalc/calibration.py`:
```python
"""Efficiency curves that turn work into time. Priors until measured."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Curve:
    points: tuple[tuple[float, float, float, float], ...]   # (x, low, mid, high)

    def __post_init__(self):
        if not self.points:
            raise ValueError("Curve needs at least one point")
        xs = [p[0] for p in self.points]
        if xs != sorted(xs) or len(set(xs)) != len(xs):
            raise ValueError("Curve x values must be strictly ascending")
        for x, lo, mid, hi in self.points:
            if not (lo <= mid <= hi):
                raise ValueError(f"Curve point at x={x} must satisfy low <= mid <= high")
            if lo <= 0:
                raise ValueError(f"Curve values must be positive, got {lo} at x={x}")

    def at(self, x: float) -> tuple[float, float, float]:
        pts = self.points
        if x <= pts[0][0]:
            return pts[0][1:]
        if x >= pts[-1][0]:
            return pts[-1][1:]
        for (x0, *a), (x1, *b) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return tuple(ya + t * (yb - ya) for ya, yb in zip(a, b))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class Calibration:
    bw_util: Curve
    mfu: Curve
    overhead_ms: tuple[float, float, float]
    source: str

    def __post_init__(self):
        lo, mid, hi = self.overhead_ms
        if not (0 <= lo <= mid <= hi):
            raise ValueError("overhead_ms must satisfy 0 <= low <= mid <= high")


def _curve(*pts: tuple[float, float, float]) -> Curve:
    """Build a Curve from (x, low, high) with mid at the midpoint."""
    return Curve(tuple((x, lo, (lo + hi) / 2, hi) for x, lo, hi in pts))


_BW_X = (1, 8, 64, 512)
_MFU_X = (1, 64, 512, 4096)


def _prior(bw: tuple, mfu: tuple, overhead: tuple[float, float]) -> Calibration:
    lo, hi = overhead
    return Calibration(
        bw_util=_curve(*((x, a, b) for x, (a, b) in zip(_BW_X, bw))),
        mfu=_curve(*((x, a, b) for x, (a, b) in zip(_MFU_X, mfu))),
        overhead_ms=(lo, (lo + hi) / 2, hi),
        source="prior",
    )


_VLLM_MFU = ((0.02, 0.05), (0.1, 0.2), (0.3, 0.45), (0.35, 0.5))

PRIORS: dict[str, Calibration] = {
    "vllm_bf16": _prior(((0.5, 0.7), (0.6, 0.8), (0.65, 0.85), (0.65, 0.85)), _VLLM_MFU, (1, 3)),
    "vllm_int4": _prior(((0.35, 0.55), (0.45, 0.65), (0.5, 0.7), (0.5, 0.7)), _VLLM_MFU, (1, 3)),
    "llamacpp_q4": _prior(((0.3, 0.5), (0.35, 0.6), (0.4, 0.65), (0.4, 0.65)),
                          ((0.01, 0.03), (0.08, 0.15), (0.2, 0.35), (0.25, 0.4)), (2, 5)),
    "hf_bf16": _prior(((0.4, 0.6), (0.5, 0.7), (0.5, 0.7), (0.5, 0.7)),
                      ((0.02, 0.05), (0.1, 0.2), (0.25, 0.4), (0.3, 0.45)), (3, 8)),
}
PRIORS["llamacpp_q8"] = PRIORS["llamacpp_q4"]
PRIORS["vllm_q4"] = PRIORS["vllm_int4"]
PRIORS["vllm_q8"] = PRIORS["vllm_bf16"]
PRIORS["hf_int4"] = PRIORS["vllm_int4"]


def get(name: str) -> Calibration:
    if name not in PRIORS:
        raise KeyError(f"unknown calibration {name!r}; built-in: {', '.join(sorted(PRIORS))}")
    return PRIORS[name]
```

- [ ] **Step 9: 통과 확인, 커밋**

Run: `uv run pytest -q && uv run ruff check .`
Expected: 전체 PASS
```bash
git add -A && git commit -m "feat: project scaffold, GPU specs, workload, quant table and calibration priors"
```

---

### Task 2: 작업량 식 (`work.py`)

**Files:**
- Create: `src/servcalc/work.py`, `tests/test_work.py`

**Interfaces:**
- Consumes: `Arch`(vramcalc), `vramcalc.memory.kv_cache`
- Produces: `Work(bytes: float, flops: float)` (덧셈 지원), `Weights(resident: int, stream: int)`, `weight_bytes(arch, bits, override=None) -> Weights`, `kv_bytes_per_token(arch, kv_bits) -> int`, `p_mm(arch)`, `p_body(arch)`, `decode_step(arch, n, sum_ctx, kv_tok, w_stream, kv_realloc) -> Work`, `prefill_chunk(arch, c_len, prior_ctx, last, kv_tok) -> Work`(가중치 제외), `prefill_chunks(prompt_len, chunk) -> list[tuple[int, int, bool]]`(`(C_j, c₀_j, last)`), `prefill_total_flops(arch, prompt_len) -> float`

- [ ] **Step 1: 실패 테스트**

`tests/test_work.py` (기대값은 tiny: P=106816, P_mm=90432, P_body=74048, P_lin=73728, W_res=213632, W_stream=180864, KV_tok=256):
```python
import pytest

from servcalc import work


def test_param_splits(tiny):
    assert work.p_mm(tiny) == 90_432          # P - V*h (untied input embedding is a lookup)
    assert work.p_body(tiny) == 74_048        # P_mm - V*h (lm_head)
    assert work.p_lin(tiny) == 73_728


def test_weight_bytes_bf16(tiny):
    w = work.weight_bytes(tiny, 16)
    assert (w.resident, w.stream) == (213_632, 180_864)


def test_weight_bytes_quantized_keeps_other_tensors_16bit(tiny):
    w = work.weight_bytes(tiny, 4)
    assert w.resident == 73_728 * 4 // 8 + (106_816 - 73_728) * 2
    assert w.stream == w.resident - 256 * 64 * 2


def test_weight_bytes_override(tiny):
    w = work.weight_bytes(tiny, 4, override=100_000)
    assert w.resident == 100_000
    assert w.stream == 100_000 - 32_768
    assert work.weight_bytes(tiny, 4, override=10).stream == 0   # clamped


def test_weight_bytes_tied(tiny_cfg):
    from vramcalc import Arch
    tied = Arch.from_config({**tiny_cfg, "tie_word_embeddings": True})
    w = work.weight_bytes(tied, 16)
    assert w.stream == w.resident


def test_kv_bytes_per_token(tiny):
    assert work.kv_bytes_per_token(tiny, 16) == 256
    assert work.kv_bytes_per_token(tiny, 8) == 128


def test_decode_step(tiny):
    # N=2, c̄=10.5: bytes = 180864 + 256*(2*10.5 + 2) = 186752; flops = 2*90432*2 + 4*2*64*2*10.5 = 372480
    w = work.decode_step(tiny, n=2, sum_ctx=21.0, kv_tok=256, w_stream=180_864, kv_realloc=False)
    assert (w.bytes, w.flops) == (186_752.0, 372_480.0)


def test_decode_step_weight_part_is_n_independent(tiny):
    a = work.decode_step(tiny, 1, 10.0, 256, 180_864, False)
    b = work.decode_step(tiny, 4, 40.0, 256, 180_864, False)
    assert b.bytes - a.bytes == 256 * (40 - 10) + 256 * 3   # only KV read + write grow


def test_decode_step_kv_realloc(tiny):
    a = work.decode_step(tiny, 2, 21.0, 256, 180_864, False)
    b = work.decode_step(tiny, 2, 21.0, 256, 180_864, True)
    assert b.bytes - a.bytes == 2 * 256 * 21.0
    assert b.flops == a.flops


def test_prefill_chunk(tiny):
    # C=4, c0=4, not last: bytes = 256*(4+4) = 2048; flops = 2*74048*4 + 4*2*64*(16 + 10) = 605696
    w = work.prefill_chunk(tiny, c_len=4, prior_ctx=4, last=False, kv_tok=256)
    assert (w.bytes, w.flops) == (2048.0, 605_696.0)
    last = work.prefill_chunk(tiny, 4, 4, True, 256)
    assert last.flops == 605_696.0 + 2 * 256 * 64


def test_prefill_chunks_and_conservation(tiny):
    assert work.prefill_chunks(8, 4) == [(4, 0, False), (4, 4, True)]
    assert work.prefill_chunks(8, 6) == [(6, 0, False), (2, 6, True)]
    assert work.prefill_chunks(8, None) == [(8, 0, True)]
    total = work.prefill_total_flops(tiny, 8)   # 2*74048*8 + 2*256*64 + 2*2*64*8*9
    assert total == 1_235_968.0
    for chunk in (1, 3, 4, 8, None):
        s = sum(work.prefill_chunk(tiny, C, c0, last, 256).flops for C, c0, last in work.prefill_chunks(8, chunk))
        assert s == total


def test_work_add():
    assert work.Work(1.0, 2.0) + work.Work(3.0, 4.0) == work.Work(4.0, 6.0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_work.py -q`
Expected: `ModuleNotFoundError: No module named 'servcalc.work'`

- [ ] **Step 3: 구현**

```python
"""Work formulas: bytes moved and FLOPs per step. Spec section 5. Engine-agnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vramcalc import Arch
from vramcalc.memory import kv_cache


@dataclass(frozen=True)
class Work:
    bytes: float
    flops: float

    def __add__(self, other: Work) -> Work:
        return Work(self.bytes + other.bytes, self.flops + other.flops)


@dataclass(frozen=True)
class Weights:
    resident: int   # bytes held in VRAM
    stream: int     # bytes read per step (untied input embedding is a lookup, not a scan)


def p_lin(arch: Arch) -> int:
    return arch.num_layers * (arch.attn_params() + arch.mlp_params())


def p_mm(arch: Arch) -> int:
    embed = arch.embed_params()
    return arch.param_count() - (0 if arch.tie_word_embeddings else embed)


def p_body(arch: Arch) -> int:
    return p_mm(arch) - arch.embed_params()


def weight_bytes(arch: Arch, bits: float, override: int | None = None) -> Weights:
    lin = p_lin(arch)
    resident = override if override is not None else math.ceil(lin * bits / 8) + (arch.param_count() - lin) * 2
    embed_bytes = 0 if arch.tie_word_embeddings else arch.embed_params() * 2
    return Weights(resident, max(0, resident - embed_bytes))


def kv_bytes_per_token(arch: Arch, kv_bits: int) -> int:
    return kv_cache(arch, 1, 1, 1) * kv_bits // 8


def decode_step(arch: Arch, n: int, sum_ctx: float, kv_tok: int, w_stream: int,
                kv_realloc: bool) -> Work:
    kv_read = kv_tok * sum_ctx
    bytes_ = w_stream + kv_read + kv_tok * n + (2 * kv_read if kv_realloc else 0)
    flops = 2 * p_mm(arch) * n + 4 * arch.num_layers * arch.q * sum_ctx
    return Work(float(bytes_), float(flops))


def prefill_chunk(arch: Arch, c_len: int, prior_ctx: int, last: bool, kv_tok: int) -> Work:
    """One prefill chunk without the weight read (callers add it once per step)."""
    L, q = arch.num_layers, arch.q
    attn = 4 * L * q * (c_len * prior_ctx + c_len * (c_len + 1) / 2)
    flops = 2 * p_body(arch) * c_len + attn + (2 * arch.embed_params() if last else 0)
    return Work(float(kv_tok * (prior_ctx + c_len)), float(flops))


def prefill_chunks(prompt_len: int, chunk: int | None) -> list[tuple[int, int, bool]]:
    """(chunk_len, prior_ctx, is_last) for each chunk of a prompt."""
    c = prompt_len if chunk is None else chunk
    k = math.ceil(prompt_len / c)
    out = []
    for j in range(k):
        start = j * c
        out.append((min(c, prompt_len - start), start, j == k - 1))
    return out


def prefill_total_flops(arch: Arch, prompt_len: int) -> float:
    i = prompt_len
    return float(2 * p_body(arch) * i + 2 * arch.embed_params()
                 + 2 * arch.num_layers * arch.q * i * (i + 1))
```

- [ ] **Step 4: 통과, 커밋**

Run: `uv run pytest -q && uv run ruff check .`
```bash
git add -A && git commit -m "feat: work formulas for decode and prefill"
```

---

### Task 3: 시간 변환 (`timing.py`)

**Files:**
- Create: `src/servcalc/timing.py`, `tests/test_timing.py`

**Interfaces:**
- Consumes: `Work`, `GpuSpec`, `Calibration`
- Produces: `Range(low, mid, high)` (덧셈, `scale(k)`, `invert(numerator)`, `weighted(a, wa, b, wb)`), `StepTime(ms: Range, bound: str)`, `step_time(work, gpu, cal, tokens) -> StepTime`

- [ ] **Step 1: 실패 테스트**

```python
import pytest

from servcalc.calibration import Calibration, Curve
from servcalc.gpus import GpuSpec
from servcalc.timing import Range, step_time
from servcalc.work import Work

GPU = GpuSpec("t", 10**9, bandwidth_gbs=1.0, bf16_tflops=0.001)   # 1 GB/s, 1 GFLOP/s
CAL = Calibration(Curve(((1, 0.25, 0.5, 1.0),)), Curve(((1, 0.25, 0.5, 1.0),)), (1.0, 1.0, 1.0), "test")


def test_memory_bound():
    # bytes 1e6 at 1 GB/s -> 1 ms at util 1.0, 2 ms at 0.5, 4 ms at 0.25; flops 1e5 at 1 GFLOP/s -> 0.1 ms
    st = step_time(Work(1e6, 1e5), GPU, CAL, tokens=1)
    assert st.bound == "memory"
    assert st.ms == Range(1.0 + 1, 2.0 + 1, 4.0 + 1)


def test_compute_bound():
    st = step_time(Work(1e5, 1e6), GPU, CAL, tokens=1)
    assert st.bound == "compute"
    assert st.ms.mid == 2.0 + 1


def test_overhead_bound():
    st = step_time(Work(1e3, 1e3), GPU, CAL, tokens=1)
    assert st.bound == "overhead"


def test_range_ops():
    r = Range(1.0, 2.0, 4.0)
    assert r + Range(1.0, 1.0, 1.0) == Range(2.0, 3.0, 5.0)
    assert r.scale(2) == Range(2.0, 4.0, 8.0)
    assert r.invert(1000) == Range(250.0, 500.0, 1000.0)           # low from high
    assert Range.weighted(Range(1, 2, 3), 0.25, Range(5, 6, 7), 0.75) == Range(4.0, 5.0, 6.0)
    with pytest.raises(ValueError, match="low <= mid <= high"):
        Range(3.0, 2.0, 1.0)
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_timing.py -q`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
"""Turn Work into time with efficiency curves. Spec section 6."""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import Calibration
from .gpus import GpuSpec
from .work import Work


@dataclass(frozen=True)
class Range:
    low: float
    mid: float
    high: float

    def __post_init__(self):
        if not (self.low <= self.mid <= self.high):
            raise ValueError(f"Range must satisfy low <= mid <= high, got {self}")

    def __add__(self, o: Range) -> Range:
        return Range(self.low + o.low, self.mid + o.mid, self.high + o.high)

    def scale(self, k: float) -> Range:
        return Range(self.low * k, self.mid * k, self.high * k)

    def invert(self, numerator: float) -> Range:
        """numerator / self, swapping ends so that low <= high still holds."""
        return Range(numerator / self.high, numerator / self.mid, numerator / self.low)

    @staticmethod
    def weighted(a: Range, wa: float, b: Range, wb: float) -> Range:
        return a.scale(wa) + b.scale(wb)


@dataclass(frozen=True)
class StepTime:
    ms: Range
    bound: str   # "memory" | "compute" | "overhead", judged at mid


def step_time(work: Work, gpu: GpuSpec, cal: Calibration, tokens: int) -> StepTime:
    bw = cal.bw_util.at(tokens)
    mfu = cal.mfu.at(tokens)
    oh = cal.overhead_ms

    def one(util_bw: float, util_mfu: float, overhead: float) -> tuple[float, float, float]:
        t_mem = work.bytes / (gpu.bandwidth_gbs * 1e9 * util_bw) * 1000
        t_comp = work.flops / (gpu.bf16_tflops * 1e12 * util_mfu) * 1000
        return t_mem, t_comp, max(t_mem, t_comp) + overhead

    # low time uses high efficiency and low overhead; high time the reverse
    _, _, lo = one(bw[2], mfu[2], oh[0])
    t_mem, t_comp, mid = one(bw[1], mfu[1], oh[1])
    _, _, hi = one(bw[0], mfu[0], oh[2])
    if oh[1] > max(t_mem, t_comp):
        bound = "overhead"
    else:
        bound = "memory" if t_mem > t_comp else "compute"
    return StepTime(Range(lo, mid, hi), bound)
```

- [ ] **Step 4: 통과, 커밋**

```bash
uv run pytest -q && uv run ruff check . && git add -A && git commit -m "feat: roofline timing with ranges"
```

---

### Task 4: 배치 모드와 용량 (`modes.py`)

**Files:**
- Create: `src/servcalc/profiles.py`(dataclass만, 팩토리는 Task 5), `src/servcalc/modes.py`, `tests/test_modes.py`, `tests/test_capacity.py`

**Interfaces:**
- Consumes: Task 2, 3
- Produces: `EngineProfile` dataclass(spec 4.1 필드, 검증), `DoesNotFitError(ValueError)`, `Capacity(usable, seq_kv_peak, memory_capacity, configured)`, `capacity(gpu, w_res, kv_tok, workload, profile) -> Capacity`, `BatchEstimate`(spec 4.3), `point(n, arch, gpu, workload, profile, cal, weights, kv_tok, cap) -> BatchEstimate` (static과 continuous를 `profile.batching`으로 분기)

- [ ] **Step 1: `EngineProfile` dataclass 작성** (`profiles.py`, 팩토리는 Task 5)

```python
"""Engine profiles: everything that differs between serving engines, as values."""

from __future__ import annotations

from dataclasses import dataclass

from . import quant as _quant

BATCHING = ("continuous", "static")


@dataclass(frozen=True)
class EngineProfile:
    name: str
    version: str
    batching: str
    max_seqs: int | None
    quant: str | None
    kv_bits: int
    kv_block: int
    kv_prealloc_ctx: int | None
    memory_budget_ratio: float
    workspace_ratio: float
    token_budget: int | None
    mixes_prefill: bool
    kv_realloc: bool
    calibration: str

    def __post_init__(self):
        if self.batching not in BATCHING:
            raise ValueError(f"batching must be one of {BATCHING}, got {self.batching!r}")
        _quant.bits(self.quant)   # validates the name
        if self.kv_bits not in (8, 16):
            raise ValueError(f"kv_bits must be 8 or 16, got {self.kv_bits!r}")
        for f in ("max_seqs", "kv_prealloc_ctx", "token_budget"):
            v = getattr(self, f)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ValueError(f"{f} must be a positive integer or None, got {v!r}")
        if self.token_budget is not None and self.token_budget < 2:
            raise ValueError("token_budget must be >= 2 (one decode token plus one prefill token)")
        if not isinstance(self.kv_block, int) or self.kv_block < 0:
            raise ValueError(f"kv_block must be a non-negative integer, got {self.kv_block!r}")
        for f in ("memory_budget_ratio", "workspace_ratio"):
            v = getattr(self, f)
            if not (0 < v <= 1):
                raise ValueError(f"{f} must be in (0, 1], got {v!r}")
        if self.memory_budget_ratio - self.workspace_ratio <= 0:
            raise ValueError("workspace_ratio must be smaller than memory_budget_ratio")
        if self.batching == "static" and self.mixes_prefill:
            raise ValueError("static batching cannot mix prefill and decode")

    @property
    def weight_bits(self) -> float:
        return _quant.bits(self.quant)
```

- [ ] **Step 2: 실패 테스트**

`tests/test_capacity.py`:
```python
import pytest

from servcalc.gpus import GpuSpec
from servcalc.modes import Capacity, DoesNotFitError, capacity
from servcalc.profiles import EngineProfile
from servcalc.workload import Workload


def prof(**kw) -> EngineProfile:
    base = dict(name="t", version="0", batching="continuous", max_seqs=None, quant=None, kv_bits=16,
                kv_block=0, kv_prealloc_ctx=None, memory_budget_ratio=1.0, workspace_ratio=0.1,
                token_budget=None, mixes_prefill=False, kv_realloc=False, calibration="vllm_bf16")
    return EngineProfile(**{**base, **kw})


GPU = GpuSpec("t", 1_000_000, 1.0, 1.0)
W = Workload(prompt_len=8, gen_len=5)


def test_capacity_basic():
    # usable = 1e6*1.0 - 213632 - 1e5 = 686368; seq_kv_peak = 256*13 = 3328 -> 206
    c = capacity(GPU, w_res=213_632, kv_tok=256, workload=W, profile=prof())
    assert c == Capacity(usable=686_368.0, seq_kv_peak=3328, memory_capacity=206, configured=206)


def test_capacity_block_rounding_and_max_seqs():
    c = capacity(GPU, 213_632, 256, W, prof(kv_block=16, max_seqs=50))
    assert c.seq_kv_peak == 256 * 16
    assert (c.memory_capacity, c.configured) == (167, 50)


def test_capacity_prealloc_ctx():
    c = capacity(GPU, 213_632, 256, W, prof(kv_prealloc_ctx=64))
    assert c.seq_kv_peak == 256 * 64
    with pytest.raises(ValueError, match="kv_prealloc_ctx"):
        capacity(GPU, 213_632, 256, W, prof(kv_prealloc_ctx=12))   # 12 < I + O = 13


def test_does_not_fit():
    with pytest.raises(DoesNotFitError, match="weights"):
        capacity(GPU, 950_000, 256, W, prof())


@pytest.mark.parametrize("kw,match", [
    ({"batching": "dynamic"}, "batching"),
    ({"quant": "bnb"}, "quant"),
    ({"kv_bits": 4}, "kv_bits"),
    ({"token_budget": 1}, "token_budget"),
    ({"memory_budget_ratio": 0.05, "workspace_ratio": 0.1}, "workspace_ratio"),
    ({"batching": "static", "mixes_prefill": True}, "static"),
])
def test_profile_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        prof(**kw)
```

`tests/test_modes.py` (수치는 tiny, I=8, O=5, N=2, B=6; D=4, c̄=10.5, λ=0.5, c=4, k=2, f=1.0, M=1, R=3):
```python
import math

import pytest

from servcalc.calibration import Calibration, Curve
from servcalc.gpus import GpuSpec
from servcalc.modes import capacity, point
from servcalc.profiles import EngineProfile
from servcalc.work import Weights
from servcalc.workload import Workload

GPU = GpuSpec("t", 1_000_000, 1.0, 0.001)
CAL = Calibration(Curve(((1, 0.5, 0.5, 0.5),)), Curve(((1, 0.5, 0.5, 0.5),)), (1.0, 1.0, 1.0), "test")
WTS = Weights(213_632, 180_864)
W = Workload(prompt_len=8, gen_len=5, concurrency=(1, 2, 4))


def prof(**kw) -> EngineProfile:
    base = dict(name="t", version="0", batching="continuous", max_seqs=None, quant=None, kv_bits=16,
                kv_block=0, kv_prealloc_ctx=None, memory_budget_ratio=1.0, workspace_ratio=0.1,
                token_budget=6, mixes_prefill=True, kv_realloc=False, calibration="vllm_bf16")
    return EngineProfile(**{**base, **kw})


def ms(bytes_, flops):
    return max(bytes_ / (1e9 * 0.5) * 1000, flops / (1e9 * 0.5) * 1000) + 1.0


def run(n, p):
    cap = capacity(GPU, WTS.resident, 256, W, p)
    return point(n, GPU, W, p, CAL, WTS, 256, cap, tiny=None)


def test_continuous_mixed_values(tiny):
    p = prof()
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert b.resident == 3                      # M = ceil(0.5 * 2) = 1
    assert b.feasible and b.infeasible_reason is None
    # s_d: bytes 186752, flops 372480 -> compute-bound 0.74496 ms + 1 ; s_mix(c=4, c0=4): 188800 / 978176
    # chunk steps: s_mix(4, 0) = 187776 B / 969984 F, s_mix(4, 4, last) = 188800 B / 1010944 F
    ttft = ms(187_776, 969_984) + ms(188_800, 1_010_944)
    assert math.isclose(b.ttft_ms.mid, ttft)
    assert math.isclose(b.itl_ms.mid, 1.0 * ttft / 2 + 0.0 * ms(186_752, 372_480))   # f = 1
    assert math.isclose(b.total_tok_s.mid, 2 * 1000 / b.itl_ms.mid)
    assert b.bound == "compute"


def test_continuous_f_weighting_tends_to_decode_only(tiny):
    # large B makes f = λI/c small
    p = prof(token_budget=10_000)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    f = 0.5 * 8 / (10_000 - 2)
    assert 0 < f < 0.01
    assert b.itl_ms.mid < ms(186_752, 372_480) * 1.05


def test_continuous_budget_infeasible(tiny):
    p = prof(token_budget=6)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    # N=4: λ = 1, N + λI = 12 > 6
    b = point(4, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert not b.feasible and b.infeasible_reason == "token_budget"
    assert b.ttft_ms is not None                 # c = 2 >= 1 so prefill is still representable
    # N=6: c = 0
    W6 = Workload(8, 5, concurrency=(6,))
    cap6 = capacity(GPU, WTS.resident, 256, W6, p)
    b6 = point(6, tiny, GPU, W6, p, CAL, WTS, 256, cap6)
    assert b6.ttft_ms is None and b6.infeasible_reason == "token_budget"
    assert math.isclose(b6.itl_ms.mid, ms(180_864 + 256 * (6 * 10.5 + 6), 2 * 90_432 * 6 + 4 * 2 * 64 * 6 * 10.5))


def test_continuous_budget_none_means_whole_prefill(tiny):
    p = prof(token_budget=None)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(4, tiny, GPU, W, p, CAL, WTS, 256, cap)     # λ = 1 -> f = min(1, λI/I) = 1
    assert b.feasible
    assert math.isclose(b.ttft_ms.mid, ms(180_864 + 256 * (4 * 10.5 + 4) + 256 * 8,
                                         2 * 90_432 * 4 + 4 * 2 * 64 * 4 * 10.5 + 1_235_968))


def test_continuous_unmixed(tiny):
    p = prof(mixes_prefill=False, token_budget=6)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    s_d = ms(186_752, 372_480)
    # chunks (6,0,False): bytes 180864 - 32768 + 256*6 = 149632, flops 899328 ; (2,6,True): 180864 + 256*8 = 182912, flops 336640
    t_p = ms(149_632, 899_328) + ms(182_912, 336_640)
    assert math.isclose(b.itl_ms.mid, s_d + 0.5 * t_p)
    assert math.isclose(b.ttft_ms.mid, t_p + 0.5 * s_d)
    assert b.resident == 2 + math.ceil(0.5 * t_p / (s_d + 0.5 * t_p))


def test_static(tiny):
    p = prof(batching="static", mixes_prefill=False, token_budget=None)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert b.resident == 2
    assert math.isclose(b.ttft_ms.mid, ms(184_960, 2_471_936))           # N*I prefill
    steps = [ms(180_864 + 256 * (2 * (9 + j) + 2), 2 * 90_432 * 2 + 4 * 2 * 64 * 2 * (9 + j)) for j in range(4)]
    assert math.isclose(b.itl_ms.mid, sum(steps) / 4)
    assert steps == sorted(steps)                                          # monotone in j


def test_max_seqs_and_memory_reasons(tiny):
    p = prof(max_seqs=2)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)                     # R = 3 > 2
    assert b.infeasible_reason == "max_seqs"
    small = GpuSpec("s", 220_000, 1.0, 0.001)                              # usable = 220000 - 213632 - 22000 < 0
    import pytest
    from servcalc.modes import DoesNotFitError
    with pytest.raises(DoesNotFitError):
        capacity(small, WTS.resident, 256, W, p)
```
`run()` 헬퍼는 사용하지 않으므로 삭제합니다(위 코드에서 제거).

- [ ] **Step 3: 실패 확인**

Run: `uv run pytest tests/test_capacity.py tests/test_modes.py -q`
Expected: `ModuleNotFoundError: No module named 'servcalc.modes'`

- [ ] **Step 4: `modes.py` 구현**

```python
"""Batching modes (static, continuous steady state) and capacity. Spec sections 5.4 and 7."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vramcalc import Arch

from . import work
from .calibration import Calibration
from .gpus import GpuSpec
from .profiles import EngineProfile
from .timing import Range, StepTime, step_time
from .work import Weights, Work
from .workload import Workload


class DoesNotFitError(ValueError):
    """Weights alone exceed the engine's memory budget; offload is not modelled."""


@dataclass(frozen=True)
class Capacity:
    usable: float
    seq_kv_peak: int
    memory_capacity: int
    configured: int


@dataclass(frozen=True)
class BatchEstimate:
    batch: int
    resident: int
    feasible: bool
    infeasible_reason: str | None
    ttft_ms: Range | None
    itl_ms: Range
    seq_tok_s: Range
    total_tok_s: Range
    bound: str


def _roundup(x: int, block: int) -> int:
    return x if block == 0 else math.ceil(x / block) * block


def capacity(gpu: GpuSpec, w_res: int, kv_tok: int, workload: Workload, profile: EngineProfile) -> Capacity:
    usable = gpu.vram_bytes * profile.memory_budget_ratio - w_res - gpu.vram_bytes * profile.workspace_ratio
    if usable <= 0:
        raise DoesNotFitError(
            f"weights ({w_res / 1e9:.2f} GB) exceed the memory budget of {gpu.name} "
            f"({gpu.vram_bytes * (profile.memory_budget_ratio - profile.workspace_ratio) / 1e9:.2f} GB usable); "
            "CPU offload is not modelled")
    total_len = workload.prompt_len + workload.gen_len
    if profile.kv_prealloc_ctx is not None:
        if profile.kv_prealloc_ctx < total_len:
            raise ValueError(
                f"kv_prealloc_ctx={profile.kv_prealloc_ctx} is shorter than prompt_len + gen_len={total_len}; "
                "truncation and context shifting are not modelled")
        seq_kv_peak = kv_tok * profile.kv_prealloc_ctx
    else:
        seq_kv_peak = kv_tok * _roundup(total_len, profile.kv_block)
    mem = int(usable // seq_kv_peak)
    conf = mem if profile.max_seqs is None else min(mem, profile.max_seqs)
    return Capacity(usable, seq_kv_peak, mem, conf)


def _prefill_ctx_bytes(kv_tok: int, prompt_len: int, profile: EngineProfile) -> int:
    ctx = profile.kv_prealloc_ctx if profile.kv_prealloc_ctx is not None else _roundup(prompt_len, profile.kv_block)
    return kv_tok * ctx


def _decode_only(arch, n, workload, kv_tok, wts, profile) -> Work:
    return work.decode_step(arch, n, n * workload.mean_decode_ctx, kv_tok, wts.stream, profile.kv_realloc)


def _finish(n, resident, reasons, ttft, itl: StepTime, cap: Capacity, kv_needed: float, profile) -> BatchEstimate:
    if kv_needed > cap.usable:
        reasons.insert(0, "memory")
    if profile.max_seqs is not None and resident > profile.max_seqs:
        reasons.insert(1 if reasons and reasons[0] == "memory" else 0, "max_seqs")
    reason = reasons[0] if reasons else None
    seq = itl.ms.invert(1000.0)
    return BatchEstimate(n, resident, reason is None, reason, ttft, itl.ms, seq, seq.scale(n), itl.bound)


def point(n: int, arch: Arch, gpu: GpuSpec, workload: Workload, profile: EngineProfile,
          cal: Calibration, wts: Weights, kv_tok: int, cap: Capacity) -> BatchEstimate:
    if profile.batching == "static":
        return _static(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap)
    return _continuous(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap)


def _static(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap) -> BatchEstimate:
    I, D = workload.prompt_len, workload.decode_steps
    pre = Work(float(wts.stream), 0.0) + work.prefill_chunk(arch, n * I, 0, True, kv_tok)
    # prefill_chunk with c_len = n*I treats the batch as one causal sequence; correct the attention term
    pre = Work(pre.bytes, pre.flops
               - 4 * arch.num_layers * arch.q * ((n * I) * (n * I + 1) / 2 - n * I * (I + 1) / 2)
               - 2 * arch.embed_params() + 2 * arch.embed_params() * n)
    ttft = step_time(pre, gpu, cal, n * I).ms
    steps = [step_time(work.decode_step(arch, n, n * (I + 1 + j), kv_tok, wts.stream, profile.kv_realloc),
                       gpu, cal, n) for j in range(D)]
    mean = Range(sum(s.ms.low for s in steps) / D, sum(s.ms.mid for s in steps) / D, sum(s.ms.high for s in steps) / D)
    bound = steps[D // 2].bound
    kv_needed = n * cap.seq_kv_peak
    return _finish(n, n, [], ttft, StepTime(mean, bound), cap, kv_needed, profile)


def _continuous(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap) -> BatchEstimate:
    I, D = workload.prompt_len, workload.decode_steps
    lam = n / D
    s_d = step_time(_decode_only(arch, n, workload, kv_tok, wts, profile), gpu, cal, n)
    reasons: list[str] = []

    if profile.mixes_prefill:
        B = profile.token_budget
        c = I if B is None else B - n
        if B is not None and (c < 1 or n + lam * I > B):
            reasons.append("token_budget")
        if c < 1:
            # prefill cannot be represented; report decode-only numbers
            kv_needed = n * cap.seq_kv_peak
            return _finish(n, n, reasons, None, s_d, cap, kv_needed, profile)
        chunks = work.prefill_chunks(I, c)
        k = len(chunks)
        f = min(1.0, lam * I / c)
        base = _decode_only(arch, n, workload, kv_tok, wts, profile)
        steps = [step_time(base + work.prefill_chunk(arch, C, c0, last, kv_tok), gpu, cal, n + C)
                 for C, c0, last in chunks]
        ttft = steps[0].ms
        for st in steps[1:]:
            ttft = ttft + st.ms
        mixed_mean = ttft.scale(1 / k)   # the mixed-step representative is the mean of the real chunk steps
        itl = StepTime(Range.weighted(mixed_mean, f, s_d.ms, 1 - f), steps[0].bound if f >= 0.5 else s_d.bound)
        m = math.ceil(lam * k)
    else:
        c = profile.token_budget
        chunks = work.prefill_chunks(I, c)
        k = len(chunks)
        t_p = None
        for C, c0, last in chunks:
            wread = wts.stream - (0 if last else (0 if arch.tie_word_embeddings else 2 * arch.embed_params()))
            st = step_time(Work(float(wread), 0.0) + work.prefill_chunk(arch, C, c0, last, kv_tok), gpu, cal, C).ms
            t_p = st if t_p is None else t_p + st
        itl_ms = s_d.ms + t_p.scale(lam)
        itl = StepTime(itl_ms, s_d.bound)
        ttft = t_p + s_d.ms.scale(0.5)
        m = math.ceil(lam * t_p.mid / itl_ms.mid)

    resident = n + m
    kv_needed = n * cap.seq_kv_peak + m * _prefill_ctx_bytes(kv_tok, I, profile)
    return _finish(n, resident, reasons, ttft, itl, cap, kv_needed, profile)
```

**static prefill 주의**: 배치 N개의 prompt는 각자 독립적으로 causal attention을 하므로 attention 항은 `N × 2·L·q·I(I+1)`이고 lm_head는 `N`번입니다. 위 코드는 `prefill_chunk`를 재사용한 뒤 보정하는 대신, 더 읽기 쉽게 직접 계산해도 됩니다:
```python
    pre = Work(float(wts.stream) + kv_tok * n * I,
               2 * work.p_body(arch) * n * I + 2 * arch.embed_params() * n
               + 2 * arch.num_layers * arch.q * n * I * (I + 1))
```
이 직접 계산 형태를 사용하고 보정 코드는 쓰지 않습니다.

- [ ] **Step 5: 통과 확인**

Run: `uv run pytest -q`
Expected: 전체 PASS. `test_continuous_unmixed`의 `wread` 기대값(149632 = 180864 − 32768 + 1536)은 untied lm_head 2·V·h = 32768을 뺀 값입니다.

- [ ] **Step 6: 커밋**

```bash
uv run ruff check . && git add -A && git commit -m "feat: static and continuous batching modes with capacity judgment"
```

---

### Task 5: 엔진 프로필 팩토리

**Files:**
- Modify: `src/servcalc/profiles.py`
- Create: `tests/test_profiles.py`

**Interfaces:**
- Produces: `vllm(chunked=True, **overrides)`, `ollama(num_parallel=4, num_ctx=2048, quant="gguf_q4_k_m", **overrides)`, `hf(**overrides)`, `BUILTIN = {"vllm": vllm, "ollama": ollama, "hf": hf}`, `resolve_profile(profile, config: dict | None) -> tuple[EngineProfile, list[str]]`(warnings)

- [ ] **Step 1: 실패 테스트**

```python
import dataclasses

import pytest

from servcalc.profiles import BUILTIN, hf, ollama, resolve_profile, vllm


def test_vllm_defaults():
    p = vllm()
    assert (p.batching, p.max_seqs, p.quant, p.kv_block, p.token_budget, p.mixes_prefill) == \
        ("continuous", 256, None, 16, 512, True)
    assert (p.memory_budget_ratio, p.workspace_ratio, p.calibration) == (0.90, 0.05, "vllm_bf16")
    assert p.version == "0.6.3"
    nc = vllm(chunked=False)
    assert nc.token_budget is None and nc.mixes_prefill is False


def test_ollama_defaults_and_args():
    p = ollama()
    assert (p.max_seqs, p.kv_prealloc_ctx, p.quant, p.token_budget, p.mixes_prefill) == (4, 2048, "gguf_q4_k_m", 512, False)
    assert p.calibration == "llamacpp_q4"
    q = ollama(num_parallel=1, num_ctx=8192, quant="gguf_q8_0")
    assert (q.max_seqs, q.kv_prealloc_ctx, q.calibration) == (1, 8192, "llamacpp_q8")


def test_hf_defaults():
    p = hf()
    assert (p.batching, p.max_seqs, p.kv_realloc, p.token_budget, p.workspace_ratio) == ("static", None, True, None, 0.20)
    assert "bf16" in p.version


def test_overrides_and_replace():
    p = vllm(kv_bits=8, max_seqs=32)
    assert (p.kv_bits, p.max_seqs) == (8, 32)
    assert dataclasses.replace(p, quant="awq").weight_bits == 4


def test_resolve_by_name_and_quant_from_config():
    cfg = {"quantization_config": {"quant_method": "awq", "bits": 4}}
    p, warns = resolve_profile("vllm", cfg)
    assert p.quant == "awq" and p.calibration == "vllm_int4"
    assert any("storage" in w for w in warns)
    p2, _ = resolve_profile(vllm(quant=None), cfg)          # explicit profile wins over config
    assert p2.quant is None
    p3, _ = resolve_profile("ollama", cfg)                   # gguf engines ignore HF quantization_config
    assert p3.quant == "gguf_q4_k_m"


def test_resolve_rejects_unknown():
    with pytest.raises(KeyError, match="vllm, ollama, hf"):
        resolve_profile("sglang", None)
    with pytest.raises(ValueError, match="bitsandbytes"):
        resolve_profile("vllm", {"quantization_config": {"quant_method": "bitsandbytes"}})
    with pytest.raises(ValueError, match="quant_method"):
        resolve_profile("vllm", {"quantization_config": {"bits": 4}})


def test_builtin_names():
    assert set(BUILTIN) == {"vllm", "ollama", "hf"}
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_profiles.py -q`
Expected: `ImportError: cannot import name 'BUILTIN'`

- [ ] **Step 3: 팩토리 구현** (`profiles.py` 끝에 추가)

```python
def _cal(engine: str, quant_name: str | None) -> str:
    return f"{engine}_{_quant.curve_suffix(quant_name)}"


def vllm(chunked: bool = True, **overrides) -> EngineProfile:
    """vLLM 0.6.3. Chunked prefill is auto-enabled for max_model_len > 32k; budget 512 includes decode."""
    quant_name = overrides.pop("quant", None)
    fields = dict(
        name="vllm", version="0.6.3", batching="continuous", max_seqs=256, quant=quant_name,
        kv_bits=16, kv_block=16, kv_prealloc_ctx=None, memory_budget_ratio=0.90, workspace_ratio=0.05,
        token_budget=512 if chunked else None, mixes_prefill=chunked, kv_realloc=False,
        calibration=_cal("vllm", quant_name),
    )
    return EngineProfile(**{**fields, **overrides})


def ollama(num_parallel: int = 4, num_ctx: int = 2048, quant: str | None = "gguf_q4_k_m",
           **overrides) -> EngineProfile:
    """ollama 0.5.x (llama.cpp server). Prefill is modelled as a separate pass; confirm in 0.2."""
    fields = dict(
        name="ollama", version="0.5.x (llama.cpp server)", batching="continuous", max_seqs=num_parallel,
        quant=quant, kv_bits=16, kv_block=0, kv_prealloc_ctx=num_ctx, memory_budget_ratio=1.0,
        workspace_ratio=0.15, token_budget=512, mixes_prefill=False, kv_realloc=False,
        calibration=_cal("llamacpp", quant),
    )
    return EngineProfile(**{**fields, **overrides})


def hf(**overrides) -> EngineProfile:
    """transformers 4.46.x generate() with the model loaded explicitly in bf16; static batching."""
    quant_name = overrides.pop("quant", None)
    fields = dict(
        name="hf", version="transformers 4.46.x, explicit bf16 load", batching="static", max_seqs=None,
        quant=quant_name, kv_bits=16, kv_block=0, kv_prealloc_ctx=None, memory_budget_ratio=1.0,
        workspace_ratio=0.20, token_budget=None, mixes_prefill=False, kv_realloc=True,
        calibration=_cal("hf", quant_name),
    )
    return EngineProfile(**{**fields, **overrides})


BUILTIN = {"vllm": vllm, "ollama": ollama, "hf": hf}
_HF_QUANT_METHODS = {"awq": "awq", "gptq": "gptq", "fp8": "fp8"}


def resolve_profile(profile, config: dict | None) -> tuple[EngineProfile, list[str]]:
    """Turn a name or EngineProfile into a concrete profile; infer quant from config only for a bare name."""
    warnings: list[str] = []
    if isinstance(profile, EngineProfile):
        p = profile
    else:
        if profile not in BUILTIN:
            raise KeyError(f"unknown profile {profile!r}; built-in: {', '.join(BUILTIN)}")
        p = BUILTIN[profile]()
        qc = (config or {}).get("quantization_config")
        if qc is not None and p.name != "ollama":
            method = qc.get("quant_method")
            if method is None:
                raise ValueError("quantization_config has no quant_method; pass quant explicitly")
            if method not in _HF_QUANT_METHODS:
                raise ValueError(f"unsupported quant_method {method!r}; supported: {', '.join(_HF_QUANT_METHODS)}")
            q = _HF_QUANT_METHODS[method]
            p = dataclasses.replace(p, quant=q, calibration=_cal(p.name, q))
    if p.quant is not None:
        warnings.append(
            f"quant={p.quant!r} sets storage bytes only; kernel speed comes from the {p.calibration} curve")
    return p, warnings
```
파일 상단에 `import dataclasses` 추가.

- [ ] **Step 4: 통과, 커밋**

```bash
uv run pytest -q && uv run ruff check . && git add -A && git commit -m "feat: vllm, ollama and hf engine profiles"
```

---

### Task 6: `estimate()`와 결과 객체

**Files:**
- Create: `src/servcalc/estimate.py`, `tests/test_estimate.py`
- Modify: `src/servcalc/__init__.py`

**Interfaces:**
- Produces: `EstimateParams`, `ServingEstimate`(spec 4.3), `estimate(model, gpu, *, workload, profile="vllm", calibration=None, weight_bytes=None) -> ServingEstimate`; `__init__`에서 `estimate, Workload, EngineProfile, Calibration, Curve, GpuSpec, Range, BatchEstimate, ServingEstimate, DoesNotFitError, profiles` export

- [ ] **Step 1: 실패 테스트**

```python
import json

import pytest

from servcalc import DoesNotFitError, GpuSpec, Workload, estimate, profiles

from .conftest import load_config

L8 = load_config("llama-3.1-8b")


def test_llama8b_3090_vllm_sanity():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512))
    assert e.weight_bytes == 8_030_261_248 * 2
    assert e.weight_stream_bytes == e.weight_bytes - 128_256 * 4096 * 2
    assert e.kv_bytes_per_token == 2 * 32 * 1024 * 2
    assert e.memory_capacity_seqs >= 1
    assert e.configured_capacity_seqs <= 256
    p1 = e.points[0]
    assert p1.batch == 1 and p1.feasible
    assert 15 <= p1.itl_ms.mid <= 40
    assert p1.itl_ms.low <= p1.itl_ms.mid <= p1.itl_ms.high
    assert p1.seq_tok_s.low <= p1.seq_tok_s.mid <= p1.seq_tok_s.high
    assert p1.bound == "memory"
    if e.saturating_batch is not None:
        idx = [p.batch for p in e.points].index(e.saturating_batch)
        assert e.points[idx].bound == "compute" and e.points[idx - 1].bound == "memory"
    assert "steady state" in " ".join(e.assumptions)
    assert e.params.calibration_source == "prior"


def test_ollama_q4_and_num_ctx():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512), profile=profiles.ollama(num_ctx=4096))
    assert e.weight_bytes < 8_030_261_248 * 2 * 0.4
    assert e.configured_capacity_seqs == min(4, e.memory_capacity_seqs)
    assert any("storage" in w for w in e.warnings)
    with pytest.raises(ValueError, match="kv_prealloc_ctx"):
        estimate(L8, "RTX 3090", workload=Workload(1024, 2048), profile="ollama")


def test_hf_static():
    e = estimate(L8, "A100 80GB", workload=Workload(512, 128, concurrency=(1, 8)), profile="hf")
    assert all(p.resident == p.batch for p in e.points)


def test_does_not_fit():
    with pytest.raises(DoesNotFitError):
        estimate(L8, "RTX 3060", workload=Workload(1024, 512))


def test_weight_bytes_override_and_gpuspec():
    g = GpuSpec("mine", 48 * 2**30, 768.0, 155.0)
    e = estimate(L8, g, workload=Workload(1024, 512), profile=profiles.ollama(), weight_bytes=4_900_000_000)
    assert e.weight_bytes == 4_900_000_000


def test_max_users_and_serialization():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512))
    assert e.max_users(itl_ms=1e9) == max(p.batch for p in e.points if p.feasible)
    assert e.max_users(itl_ms=0.001) == 0
    d = json.loads(json.dumps(e.to_dict()))
    assert d["points"][0]["itl_ms"]["mid"] == e.points[0].itl_ms.mid
    assert d["params"]["profile"]["name"] == "vllm"
    assert "weights" in e.table() and "tok/s" in e.table()


@pytest.mark.parametrize("name", ["llama-2-7b", "llama-3.2-1b", "llama-3.2-3b", "mistral-7b-v0.3",
                                  "qwen2.5-7b", "qwen2.5-1.5b", "gemma-2-2b"])
def test_all_configs_run(name):
    e = estimate(load_config(name), "A100 80GB", workload=Workload(2048, 256, concurrency=(1, 16)))
    for p in e.points:
        assert p.itl_ms.low <= p.itl_ms.mid <= p.itl_ms.high
        assert p.total_tok_s.low <= p.total_tok_s.mid <= p.total_tok_s.high
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_estimate.py -q`
Expected: `ImportError: cannot import name 'estimate' from 'servcalc'`

- [ ] **Step 3: 구현**

`src/servcalc/estimate.py`:
```python
"""Compose work, timing, modes and profiles into a ServingEstimate."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from vramcalc import Arch
from vramcalc.loaders import load_arch, load_config

from . import work
from .calibration import Calibration, get as get_calibration
from .gpus import GpuSpec, resolve as resolve_gpu
from .modes import BatchEstimate, DoesNotFitError, capacity, point
from .profiles import EngineProfile, resolve_profile
from .timing import Range
from .workload import Workload

GB = 10**9

ASSUMPTIONS = (
    "steady state; admission queueing excluded",
    "equal request lengths (prompt_len, gen_len) for every request",
    "prefix caching off; no speculative decoding",
    "FCFS prefill scheduling with leftover token budget passed to the next request",
    "total_tok_s counts decode tokens only (the first token comes from prefill)",
)


@dataclass(frozen=True)
class EstimateParams:
    arch: Arch
    gpu: GpuSpec
    workload: Workload
    profile: EngineProfile
    calibration_source: str


@dataclass(frozen=True)
class ServingEstimate:
    memory_capacity_seqs: int
    configured_capacity_seqs: int
    saturating_batch: int | None
    points: tuple[BatchEstimate, ...]
    weight_bytes: int
    weight_stream_bytes: int
    kv_bytes_per_token: int
    params: EstimateParams
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]

    def max_users(self, itl_ms: float) -> int:
        ok = [p.batch for p in self.points if p.feasible and p.itl_ms.mid <= itl_ms]
        return max(ok) if ok else 0

    def to_dict(self) -> dict:
        def rng(r: Range | None):
            return None if r is None else dataclasses.asdict(r)
        return {
            "memory_capacity_seqs": self.memory_capacity_seqs,
            "configured_capacity_seqs": self.configured_capacity_seqs,
            "saturating_batch": self.saturating_batch,
            "weight_bytes": self.weight_bytes,
            "weight_stream_bytes": self.weight_stream_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "points": [
                {**{k: v for k, v in dataclasses.asdict(p).items() if not isinstance(v, dict)},
                 "ttft_ms": rng(p.ttft_ms), "itl_ms": rng(p.itl_ms),
                 "seq_tok_s": rng(p.seq_tok_s), "total_tok_s": rng(p.total_tok_s)}
                for p in self.points
            ],
            "params": {
                "arch": dataclasses.asdict(self.params.arch),
                "gpu": dataclasses.asdict(self.params.gpu),
                "workload": dataclasses.asdict(self.params.workload),
                "profile": dataclasses.asdict(self.params.profile),
                "calibration_source": self.params.calibration_source,
            },
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }

    def table(self) -> str:
        p = self.params
        head = [
            f"{p.arch.name or 'model'} on {p.gpu.name} via {p.profile.name} ({p.profile.version})",
            f"weights {self.weight_bytes / GB:.2f} GB resident, {self.weight_stream_bytes / GB:.2f} GB streamed per step; "
            f"KV {self.kv_bytes_per_token / 1e3:.1f} KB/token",
            f"capacity: memory {self.memory_capacity_seqs} seqs, configured {self.configured_capacity_seqs} seqs"
            + (f"; saturating batch {self.saturating_batch}" if self.saturating_batch else ""),
            f"calibration: {p.calibration_source} (ranges are low~high)",
            "",
            f"{'N':>5} {'R':>5} {'ok':<13} {'TTFT ms':>22} {'ITL ms':>20} {'seq tok/s':>18} {'total tok/s':>20} bound",
        ]
        rows = []
        for b in self.points:
            ok = "yes" if b.feasible else f"no:{b.infeasible_reason}"
            ttft = "-" if b.ttft_ms is None else f"{b.ttft_ms.mid:8.0f} [{b.ttft_ms.low:.0f}~{b.ttft_ms.high:.0f}]"
            rows.append(
                f"{b.batch:>5} {b.resident:>5} {ok:<13} {ttft:>22} "
                f"{b.itl_ms.mid:7.1f} [{b.itl_ms.low:.1f}~{b.itl_ms.high:.1f}]".ljust(0)
                + f" {b.seq_tok_s.mid:7.1f} [{b.seq_tok_s.low:.0f}~{b.seq_tok_s.high:.0f}]"
                + f" {b.total_tok_s.mid:8.0f} [{b.total_tok_s.low:.0f}~{b.total_tok_s.high:.0f}] {b.bound}")
        return "\n".join(head + rows)


def _saturating(points: tuple[BatchEstimate, ...]) -> int | None:
    prev = None
    for p in points:
        if prev == "memory" and p.bound == "compute":
            return p.batch
        if p.bound != "overhead":
            prev = p.bound
    return None


def estimate(model, gpu, *, workload: Workload, profile: EngineProfile | str = "vllm",
             calibration: Calibration | None = None, weight_bytes: int | None = None) -> ServingEstimate:
    config = load_config(model) if not isinstance(model, Arch) else None
    arch = load_arch(model)
    gpu_spec = resolve_gpu(gpu)
    prof, warnings = resolve_profile(profile, config)
    cal = calibration if calibration is not None else get_calibration(prof.calibration)
    if weight_bytes is not None and (not isinstance(weight_bytes, int) or weight_bytes <= 0):
        raise ValueError(f"weight_bytes must be a positive integer, got {weight_bytes!r}")
    wts = work.weight_bytes(arch, prof.weight_bits, weight_bytes)
    if weight_bytes is None and prof.quant is not None:
        warnings.append("quantized weight bytes approximate non-linear tensors as 16-bit; pass weight_bytes to override")
    kv_tok = work.kv_bytes_per_token(arch, prof.kv_bits)
    cap = capacity(gpu_spec, wts.resident, kv_tok, workload, prof)
    points = tuple(point(n, arch, gpu_spec, workload, prof, cal, wts, kv_tok, cap) for n in workload.concurrency)
    return ServingEstimate(
        memory_capacity_seqs=cap.memory_capacity,
        configured_capacity_seqs=cap.configured,
        saturating_batch=_saturating(points),
        points=points,
        weight_bytes=wts.resident,
        weight_stream_bytes=wts.stream,
        kv_bytes_per_token=kv_tok,
        params=EstimateParams(arch, gpu_spec, workload, prof, cal.source),
        assumptions=ASSUMPTIONS,
        warnings=tuple(warnings),
    )
```

`src/servcalc/__init__.py`:
```python
"""servcalc: predict single-GPU LLM serving capacity from config.json."""

from . import profiles
from .calibration import Calibration, Curve
from .estimate import EstimateParams, ServingEstimate, estimate
from .gpus import GpuSpec
from .modes import BatchEstimate, DoesNotFitError
from .profiles import EngineProfile
from .timing import Range
from .workload import Workload

__version__ = "0.1.0"
__all__ = ["BatchEstimate", "Calibration", "Curve", "DoesNotFitError", "EngineProfile", "EstimateParams",
           "GpuSpec", "Range", "ServingEstimate", "Workload", "estimate", "profiles"]
```

- [ ] **Step 4: 통과 확인**

Run: `uv run pytest -q`
Expected: 전체 PASS. `test_llama8b_3090_vllm_sanity`의 ITL 15~40ms는 설계 단계 손계산(25ms)과 같은 식이므로 통과해야 합니다. 실패하면 식이 아니라 사전 곡선 값을 먼저 의심합니다.

- [ ] **Step 5: 커밋**

```bash
uv run ruff check . && git add -A && git commit -m "feat: estimate() with ServingEstimate result object"
```

---

### Task 7: CLI, README, CI, 배포

**Files:**
- Create: `src/servcalc/cli.py`, `tests/test_cli.py`, `.github/workflows/ci.yml`, `CHANGELOG.md`, `LICENSE`
- Modify: `README.md`

- [ ] **Step 1: 실패 테스트**

```python
import json
import subprocess
import sys

from .conftest import CONFIG_DIR

CFG = str(CONFIG_DIR / "llama-3.1-8b.json")


def run(*args):
    return subprocess.run([sys.executable, "-m", "servcalc.cli", *args], capture_output=True, text=True, check=False)


def test_table_output():
    r = run(CFG, "--gpu", "RTX 3090")
    assert r.returncode == 0, r.stderr
    assert "capacity:" in r.stdout and "tok/s" in r.stdout


def test_json_and_slo():
    r = run(CFG, "--gpu", "A100 80GB", "--engine", "vllm", "--prompt", "2048", "--gen", "256",
            "--concurrency", "1,8,64", "--slo-itl", "50", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert [p["batch"] for p in d["points"]] == [1, 8, 64]
    assert "max_users" in d and isinstance(d["max_users"], int)


def test_ollama_options():
    r = run(CFG, "--gpu", "3090", "--engine", "ollama", "--max-seqs", "2", "--num-ctx", "4096", "--quant", "gguf_q8_0", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["params"]["profile"]["max_seqs"] == 2
    assert d["params"]["profile"]["quant"] == "gguf_q8_0"


def test_does_not_fit_exit_1():
    r = run(CFG, "--gpu", "RTX 3060")
    assert r.returncode == 1
    assert "exceed" in r.stderr and "Traceback" not in r.stderr


def test_bad_args_exit_2():
    assert run(CFG, "--gpu", "24").returncode == 2
    assert run(CFG, "--gpu", "RTX 3090", "--gen", "1").returncode == 2
    assert run(CFG, "--gpu", "RTX 3090", "--engine", "sglang").returncode == 2


def test_gpus_subcommand():
    r = run("gpus")
    assert r.returncode == 0 and "RTX A6000" in r.stdout and "GB/s" in r.stdout
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_cli.py -q`
Expected: `No module named servcalc.cli`

- [ ] **Step 3: 구현**

```python
"""Command-line interface: `servcalc MODEL --gpu NAME ...` and `servcalc gpus`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from vramcalc import UnsupportedModelError

from . import profiles
from .estimate import estimate
from .gpus import SPECS
from .modes import DoesNotFitError
from .workload import Workload

from vramcalc import gpus as _vg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="servcalc", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    est = sub.add_parser("estimate", help="predict serving capacity (default when MODEL is given)")
    est.add_argument("model")
    est.add_argument("--gpu", required=True, help="GPU name from `servcalc gpus`")
    est.add_argument("--engine", default="vllm", choices=sorted(profiles.BUILTIN))
    est.add_argument("--prompt", type=int, default=1024)
    est.add_argument("--gen", type=int, default=512)
    est.add_argument("--concurrency", default="1,2,4,8,16,32,64,128,256")
    est.add_argument("--quant")
    est.add_argument("--kv-bits", type=int)
    est.add_argument("--max-seqs", type=int)
    est.add_argument("--num-ctx", type=int, help="ollama num_ctx")
    est.add_argument("--token-budget", type=int)
    est.add_argument("--no-chunked", action="store_true", help="vllm: disable chunked prefill")
    est.add_argument("--slo-itl", type=float, metavar="MS", help="report max users whose mean ITL is within MS")
    est.add_argument("--json", action="store_true")
    sub.add_parser("gpus", help="list GPUs with bandwidth and bf16 TFLOPs")
    return parser


def _profile(a: argparse.Namespace):
    kwargs = {}
    if a.engine == "vllm":
        p = profiles.vllm(chunked=not a.no_chunked, **({"quant": a.quant} if a.quant else {}))
    elif a.engine == "ollama":
        if a.max_seqs is not None:
            kwargs["num_parallel"] = a.max_seqs
        if a.num_ctx is not None:
            kwargs["num_ctx"] = a.num_ctx
        if a.quant is not None:
            kwargs["quant"] = a.quant
        p = profiles.ollama(**kwargs)
    else:
        p = profiles.hf(**({"quant": a.quant} if a.quant else {}))
    overrides = {}
    if a.kv_bits is not None:
        overrides["kv_bits"] = a.kv_bits
    if a.max_seqs is not None and a.engine != "ollama":
        overrides["max_seqs"] = a.max_seqs
    if a.token_budget is not None:
        overrides["token_budget"] = a.token_budget
    # a bare --engine name should still infer quant from the config; only wrap when the user changed something
    if not overrides and a.quant is None and a.max_seqs is None and a.num_ctx is None and not a.no_chunked:
        return a.engine
    return dataclasses.replace(p, **overrides)


def cmd_estimate(a: argparse.Namespace) -> int:
    conc = tuple(int(x) for x in a.concurrency.split(","))
    e = estimate(a.model, a.gpu, workload=Workload(a.prompt, a.gen, conc), profile=_profile(a))
    if a.json:
        d = e.to_dict()
        if a.slo_itl is not None:
            d["max_users"] = e.max_users(a.slo_itl)
        print(json.dumps(d, indent=2))
    else:
        print(e.table())
        if a.slo_itl is not None:
            print(f"\nmax users with mean ITL <= {a.slo_itl:g} ms: {e.max_users(a.slo_itl)}")
        for w in e.warnings:
            print(f"warning: {w}")
    return 0


def cmd_gpus(_: argparse.Namespace) -> int:
    print(f"{'GPU':<20} {'VRAM GiB':>9} {'GB/s':>7} {'bf16 TFLOPs':>12}")
    for name, vram in _vg.GPUS.items():
        bw, tf = SPECS[name]
        print(f"{name:<20} {vram / 2**30:>9.0f} {bw:>7.0f} {tf:>12.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("gpus", "estimate") and not argv[0].startswith("-"):
        argv.insert(0, "estimate")
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.command is None:
        parser.print_help()
        return 2
    try:
        return {"estimate": cmd_estimate, "gpus": cmd_gpus}[a.command](a)
    except DoesNotFitError as exc:
        print(f"servcalc: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, FileNotFoundError, ImportError, UnsupportedModelError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        print(f"servcalc: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: CI, LICENSE, CHANGELOG, README**

`.github/workflows/ci.yml`은 vramcalc의 것과 동일(`uv sync --extra dev`, `ruff check .`, `pytest`, 태그 push 시 publish). LICENSE는 MIT, `Copyright (c) 2026 Seonho Hong`. CHANGELOG:
```markdown
# Changelog

## 0.1.0 (unreleased)

- Roofline estimate of concurrency, tok/s, TTFT and ITL per batch size for dense decoder-only models on one GPU
- Engine profiles: vllm (chunked prefill, paged KV), ollama (GGUF, preallocated context), hf (static generate)
- Ranges from prior efficiency curves; no measured calibration yet
- CLI: `servcalc MODEL --gpu NAME`, `servcalc gpus`
```

README는 vramcalc와 같은 구성으로 영어 먼저, 한국어 아래. 내용: 한 줄 설명, 예제(Python과 CLI 출력 표 하나), 설치, 왜 만들었는가(vramcalc의 다음 질문), 지원 범위와 1차 제외, 계산 구조(작업량 / 효율 곡선 / 모드 / 프로필 네 문단), **"모든 숫자는 실측 전 사전 범위이며 정확도 수치를 주장하지 않는다"** 명시, 0.2 계획(RunPod 실측), 한계, 라이선스. 예제 출력은 `uv run servcalc tests/configs/llama-3.1-8b.json --gpu "RTX 3090" --concurrency 1,4,16,64 --slo-itl 50`의 실제 결과를 붙입니다.

- [ ] **Step 5: 통과, 커밋**

```bash
uv run pytest -q && uv run ruff check . && git add -A && git commit -m "feat: CLI, README, CI and release metadata"
```

- [ ] **Step 6: 리뷰 후 배포**

전체 브랜치 리뷰(Codex 포함)를 받고 Critical/Important를 고친 뒤:
```bash
gh repo create sacom123/servcalc --public --source . --remote origin --push \
  --description "Predict single-GPU LLM serving capacity (concurrency, tok/s, TTFT, ITL) from config.json with a roofline model"
```
PyPI pending publisher 등록(project `servcalc`, owner `sacom123`, repo `servcalc`, workflow `ci.yml`, environment `pypi`)은 사용자가 수행합니다. 이후 CHANGELOG 날짜, `git tag v0.1.0 && git push origin main --tags`.
