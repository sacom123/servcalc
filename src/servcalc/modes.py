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


def capacity(gpu: GpuSpec, w_res: int, kv_tok: int, workload: Workload,
             profile: EngineProfile) -> Capacity:
    budget = gpu.vram_bytes * (profile.memory_budget_ratio - profile.workspace_ratio)
    usable = budget - w_res
    if usable <= 0:
        raise DoesNotFitError(
            f"weights ({w_res / 1e9:.2f} GB) exceed the memory budget of {gpu.name} "
            f"({budget / 1e9:.2f} GB usable); CPU offload is not modelled")
    total_len = workload.prompt_len + workload.gen_len
    if profile.kv_prealloc_ctx is not None:
        if profile.kv_prealloc_ctx < total_len:
            raise ValueError(
                f"kv_prealloc_ctx={profile.kv_prealloc_ctx} is shorter than "
                f"prompt_len + gen_len={total_len}; truncation and context shifting "
                "are not modelled")
        seq_kv_peak = kv_tok * profile.kv_prealloc_ctx
    else:
        seq_kv_peak = kv_tok * _roundup(total_len, profile.kv_block)
    mem = int(usable // seq_kv_peak)
    conf = mem if profile.max_seqs is None else min(mem, profile.max_seqs)
    return Capacity(usable, seq_kv_peak, mem, conf)


def _prefill_ctx_bytes(kv_tok: int, prompt_len: int, profile: EngineProfile) -> int:
    if profile.kv_prealloc_ctx is not None:
        return kv_tok * profile.kv_prealloc_ctx
    return kv_tok * _roundup(prompt_len, profile.kv_block)


def _decode_only(arch, n, workload, kv_tok, wts, profile) -> Work:
    return work.decode_step(arch, n, n * workload.mean_decode_ctx, kv_tok, wts.stream,
                            profile.kv_realloc)


def _mean(ranges: list[Range]) -> Range:
    k = len(ranges)
    return Range(sum(r.low for r in ranges) / k, sum(r.mid for r in ranges) / k,
                 sum(r.high for r in ranges) / k)


def _sum(ranges: list[Range]) -> Range:
    total = ranges[0]
    for r in ranges[1:]:
        total = total + r
    return total


def _finish(n, resident, reasons: list[str], ttft, itl: StepTime, cap: Capacity,
            kv_needed: float, profile) -> BatchEstimate:
    # reason priority: memory, max_seqs, token_budget
    if profile.max_seqs is not None and resident > profile.max_seqs:
        reasons.insert(0, "max_seqs")
    if kv_needed > cap.usable:
        reasons.insert(0, "memory")
    reason = reasons[0] if reasons else None
    seq = itl.ms.invert(1000.0)
    return BatchEstimate(n, resident, reason is None, reason, ttft, itl.ms, seq, seq.scale(n),
                         itl.bound)


def point(n: int, arch: Arch, gpu: GpuSpec, workload: Workload, profile: EngineProfile,
          cal: Calibration, wts: Weights, kv_tok: int, cap: Capacity) -> BatchEstimate:
    if profile.batching == "static":
        return _static(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap)
    return _continuous(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap)


def _static(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap) -> BatchEstimate:
    plen, d = workload.prompt_len, workload.decode_steps
    # n independent causal prompts in one forward pass: lm_head once per sequence
    pre = Work(float(wts.stream) + kv_tok * n * plen,
               2 * work.p_body(arch) * n * plen + 2 * arch.embed_params() * n
               + 2 * arch.num_layers * arch.q * n * plen * (plen + 1))
    ttft = step_time(pre, gpu, cal, n * plen).ms
    steps = [step_time(work.decode_step(arch, n, n * (plen + 1 + j), kv_tok, wts.stream,
                                        profile.kv_realloc), gpu, cal, n)
             for j in range(d)]
    itl = StepTime(_mean([s.ms for s in steps]), steps[d // 2].bound)
    return _finish(n, n, [], ttft, itl, cap, n * cap.seq_kv_peak, profile)


def _continuous(n, arch, gpu, workload, profile, cal, wts, kv_tok, cap) -> BatchEstimate:
    plen = workload.prompt_len
    lam = n / workload.decode_steps
    base = _decode_only(arch, n, workload, kv_tok, wts, profile)
    s_d = step_time(base, gpu, cal, n)
    reasons: list[str] = []

    if profile.mixes_prefill:
        budget = profile.token_budget
        c = plen if budget is None else budget - n
        if budget is not None and (c < 1 or n + lam * plen > budget):
            reasons.append("token_budget")
        if c < 1:
            # prefill cannot be represented; report decode-only numbers
            return _finish(n, n, reasons, None, s_d, cap, n * cap.seq_kv_peak, profile)
        chunks = work.prefill_chunks(plen, c)
        k = len(chunks)
        f = min(1.0, lam * plen / c)
        steps = [step_time(base + work.prefill_chunk(arch, clen, c0, last, kv_tok), gpu, cal,
                           n + clen)
                 for clen, c0, last in chunks]
        ttft = _sum([s.ms for s in steps])
        mixed_mean = ttft.scale(1 / k)   # representative mixed step: mean of the real chunk steps
        itl = StepTime(Range.weighted(mixed_mean, f, s_d.ms, 1 - f),
                       steps[0].bound if f >= 0.5 else s_d.bound)
        m = math.ceil(lam * k)
    else:
        chunks = work.prefill_chunks(plen, profile.token_budget)
        lm_head = 0 if arch.tie_word_embeddings else 2 * arch.embed_params()
        parts = []
        for clen, c0, last in chunks:
            wread = wts.stream - (0 if last else lm_head)
            parts.append(step_time(Work(float(wread), 0.0)
                                   + work.prefill_chunk(arch, clen, c0, last, kv_tok),
                                   gpu, cal, clen).ms)
        t_p = _sum(parts)
        itl = StepTime(s_d.ms + t_p.scale(lam), s_d.bound)
        ttft = t_p + s_d.ms.scale(0.5)
        m = math.ceil(lam * t_p.mid / itl.ms.mid)

    kv_needed = n * cap.seq_kv_peak + m * _prefill_ctx_bytes(kv_tok, plen, profile)
    return _finish(n, n + m, reasons, ttft, itl, cap, kv_needed, profile)
