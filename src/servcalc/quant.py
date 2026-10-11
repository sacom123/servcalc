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
