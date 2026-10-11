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
    "RTX 4070 Ti Super": (672, 88), "RTX 4080": (717, 97), "RTX 4080 Super": (736, 104),
    "RTX 4090": (1008, 165),
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
            if (not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(v) or v <= 0):
                raise ValueError(f"{f} must be a positive finite number, got {v!r}")


def resolve(gpu) -> GpuSpec:
    if isinstance(gpu, GpuSpec):
        return gpu
    if not isinstance(gpu, str):
        raise ValueError(  # noqa: TRY004
            "gpu must be a name from the built-in table or a GpuSpec with bandwidth and TFLOPs; "
            f"a bare number ({gpu!r}) cannot be timed")
    name, vram = _vg.lookup(gpu)
    bw, tf = SPECS[name]
    return GpuSpec(name, vram, float(bw), float(tf))
