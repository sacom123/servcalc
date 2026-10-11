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
    if override is not None:
        resident = override
    else:
        resident = math.ceil(lin * bits / 8) + (arch.param_count() - lin) * 2
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
    layers, q = arch.num_layers, arch.q
    attn = 4 * layers * q * (c_len * prior_ctx + c_len * (c_len + 1) / 2)
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
