"""Efficiency curves that turn work into time. Priors until measured."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


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
        for (x0, *a), (x1, *b) in pairwise(pts):
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
