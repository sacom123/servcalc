"""Compose work, timing, modes and profiles into a ServingEstimate."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from vramcalc import Arch
from vramcalc.loaders import load_arch, load_config

from . import work
from .calibration import Calibration
from .calibration import get as get_calibration
from .gpus import GpuSpec
from .gpus import resolve as resolve_gpu
from .modes import BatchEstimate, capacity, point
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


def _rng(r: Range | None) -> dict | None:
    return None if r is None else dataclasses.asdict(r)


def _fmt(r: Range | None, digits: int = 0) -> str:
    if r is None:
        return "-"
    return f"{r.mid:.{digits}f} [{r.low:.{digits}f}~{r.high:.{digits}f}]"


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
        points = []
        for p in self.points:
            d = dataclasses.asdict(p)
            for k in ("ttft_ms", "itl_ms", "seq_tok_s", "total_tok_s"):
                d[k] = _rng(getattr(p, k))
            points.append(d)
        return {
            "memory_capacity_seqs": self.memory_capacity_seqs,
            "configured_capacity_seqs": self.configured_capacity_seqs,
            "saturating_batch": self.saturating_batch,
            "weight_bytes": self.weight_bytes,
            "weight_stream_bytes": self.weight_stream_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "points": points,
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
        sat = f"; saturating batch {self.saturating_batch}" if self.saturating_batch else ""
        head = [
            f"{p.arch.name or 'model'} on {p.gpu.name} via {p.profile.name} ({p.profile.version})",
            (f"weights {self.weight_bytes / GB:.2f} GB resident, "
             f"{self.weight_stream_bytes / GB:.2f} GB streamed per step; "
             f"KV {self.kv_bytes_per_token / 1e3:.1f} KB/token"),
            (f"capacity: memory {self.memory_capacity_seqs} seqs, "
             f"configured {self.configured_capacity_seqs} seqs{sat}"),
            f"calibration: {p.calibration_source} (ranges are low~high)",
            "",
            (f"{'N':>5} {'R':>5} {'ok':<16} {'TTFT ms':>22} {'ITL ms':>22} "
             f"{'seq tok/s':>20} {'total tok/s':>22} bound"),
        ]
        rows = [
            f"{b.batch:>5} {b.resident:>5} "
            f"{('yes' if b.feasible else 'no:' + str(b.infeasible_reason)):<16} "
            f"{_fmt(b.ttft_ms):>22} {_fmt(b.itl_ms, 1):>22} "
            f"{_fmt(b.seq_tok_s, 1):>20} {_fmt(b.total_tok_s):>22} {b.bound}"
            for b in self.points
        ]
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
             calibration: Calibration | None = None,
             weight_bytes: int | None = None) -> ServingEstimate:
    config = None if isinstance(model, Arch) else load_config(model)
    arch = load_arch(model)
    gpu_spec = resolve_gpu(gpu)
    prof, warnings = resolve_profile(profile, config)
    cal = calibration if calibration is not None else get_calibration(prof.calibration)
    if weight_bytes is not None and (
            not isinstance(weight_bytes, int) or isinstance(weight_bytes, bool)
            or weight_bytes <= 0):
        raise ValueError(f"weight_bytes must be a positive integer, got {weight_bytes!r}")
    wts = work.weight_bytes(arch, prof.weight_bits, weight_bytes)
    if weight_bytes is None and prof.quant is not None:
        warnings.append("quantized weight bytes approximate non-linear tensors as 16-bit; "
                        "pass weight_bytes to override")
    kv_tok = work.kv_bytes_per_token(arch, prof.kv_bits)
    cap = capacity(gpu_spec, wts.resident, kv_tok, workload, prof)
    points = tuple(point(n, arch, gpu_spec, workload, prof, cal, wts, kv_tok, cap)
                   for n in workload.concurrency)
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
