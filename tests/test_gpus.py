import pytest
from vramcalc import gpus as vg

from servcalc.gpus import SPECS, GpuSpec, resolve


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
