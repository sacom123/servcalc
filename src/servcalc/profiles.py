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
