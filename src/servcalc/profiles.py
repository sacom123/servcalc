"""Engine profiles: everything that differs between serving engines, as values."""

from __future__ import annotations

import dataclasses
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
            raise ValueError(
                "token_budget must be >= 2 (one decode token plus one prefill token)")
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


def _cal(engine: str, quant_name: str | None) -> str:
    return f"{engine}_{_quant.curve_suffix(quant_name)}"


def vllm(chunked: bool = True, **overrides) -> EngineProfile:
    """vLLM 0.6.3. Chunked prefill with a 512-token budget that includes decode tokens."""
    quant_name = overrides.pop("quant", None)
    fields = {
        "name": "vllm", "version": "0.6.3", "batching": "continuous", "max_seqs": 256,
        "quant": quant_name, "kv_bits": 16, "kv_block": 16, "kv_prealloc_ctx": None,
        "memory_budget_ratio": 0.90, "workspace_ratio": 0.05,
        "token_budget": 512 if chunked else None, "mixes_prefill": chunked, "kv_realloc": False,
        "calibration": _cal("vllm", quant_name),
    }
    return EngineProfile(**{**fields, **overrides})


def ollama(num_parallel: int = 4, num_ctx: int = 2048, quant: str | None = "gguf_q4_k_m",
           **overrides) -> EngineProfile:
    """ollama 0.5.x (llama.cpp server). Prefill is modelled as a separate pass; confirm in 0.2."""
    fields = {
        "name": "ollama", "version": "0.5.x (llama.cpp server)", "batching": "continuous",
        "max_seqs": num_parallel, "quant": quant, "kv_bits": 16, "kv_block": 0,
        "kv_prealloc_ctx": num_ctx, "memory_budget_ratio": 1.0, "workspace_ratio": 0.15,
        "token_budget": 512, "mixes_prefill": False, "kv_realloc": False,
        "calibration": _cal("llamacpp", quant),
    }
    return EngineProfile(**{**fields, **overrides})


def hf(**overrides) -> EngineProfile:
    """transformers 4.46.x generate() with the model loaded explicitly in bf16; static batching."""
    quant_name = overrides.pop("quant", None)
    fields = {
        "name": "hf", "version": "transformers 4.46.x, explicit bf16 load", "batching": "static",
        "max_seqs": None, "quant": quant_name, "kv_bits": 16, "kv_block": 0,
        "kv_prealloc_ctx": None, "memory_budget_ratio": 1.0, "workspace_ratio": 0.20,
        "token_budget": None, "mixes_prefill": False, "kv_realloc": True,
        "calibration": _cal("hf", quant_name),
    }
    return EngineProfile(**{**fields, **overrides})


BUILTIN = {"vllm": vllm, "ollama": ollama, "hf": hf}
_HF_QUANT_METHODS = {"awq": "awq", "gptq": "gptq", "fp8": "fp8"}


def resolve_profile(profile, config: dict | None) -> tuple[EngineProfile, list[str]]:
    """Name or EngineProfile into a concrete profile; quant is read from config only for a name."""
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
                raise ValueError(f"unsupported quant_method {method!r}; "
                                 f"supported: {', '.join(_HF_QUANT_METHODS)}")
            q = _HF_QUANT_METHODS[method]
            p = dataclasses.replace(p, quant=q, calibration=_cal(p.name, q))
    if p.quant is not None:
        warnings.append(f"quant={p.quant!r} sets storage bytes only; "
                        f"kernel speed comes from the {p.calibration} curve")
    return p, warnings
