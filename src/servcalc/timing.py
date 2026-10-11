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
