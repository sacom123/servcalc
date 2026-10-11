import pytest

from servcalc.gpus import GpuSpec
from servcalc.modes import Capacity, DoesNotFitError, capacity
from servcalc.profiles import EngineProfile
from servcalc.workload import Workload


def prof(**kw) -> EngineProfile:
    base = {"name": "t", "version": "0", "batching": "continuous", "max_seqs": None, "quant": None,
                "kv_bits": 16, "kv_block": 0, "kv_prealloc_ctx": None, "memory_budget_ratio": 1.0,
                "workspace_ratio": 0.1, "token_budget": None, "mixes_prefill": False, "kv_realloc": False,
                "calibration": "vllm_bf16"}
    return EngineProfile(**{**base, **kw})


GPU = GpuSpec("t", 1_000_000, 1.0, 1.0)
W = Workload(prompt_len=8, gen_len=5)


def test_capacity_basic():
    # usable = 1e6*1.0 - 213632 - 1e5 = 686368; seq_kv_peak = 256*13 = 3328 -> 206
    c = capacity(GPU, w_res=213_632, kv_tok=256, workload=W, profile=prof())
    assert c == Capacity(usable=686_368.0, seq_kv_peak=3328, memory_capacity=206, configured=206)


def test_capacity_block_rounding_and_max_seqs():
    c = capacity(GPU, 213_632, 256, W, prof(kv_block=16, max_seqs=50))
    assert c.seq_kv_peak == 256 * 16
    assert (c.memory_capacity, c.configured) == (167, 50)


def test_capacity_prealloc_ctx():
    c = capacity(GPU, 213_632, 256, W, prof(kv_prealloc_ctx=64))
    assert c.seq_kv_peak == 256 * 64
    with pytest.raises(ValueError, match="kv_prealloc_ctx"):
        capacity(GPU, 213_632, 256, W, prof(kv_prealloc_ctx=12))   # 12 < I + O = 13


def test_does_not_fit():
    with pytest.raises(DoesNotFitError, match="weights"):
        capacity(GPU, 950_000, 256, W, prof())


@pytest.mark.parametrize("kw,match", [
    ({"batching": "dynamic"}, "batching"),
    ({"quant": "bnb"}, "quant"),
    ({"kv_bits": 4}, "kv_bits"),
    ({"token_budget": 1}, "token_budget"),
    ({"memory_budget_ratio": 0.05, "workspace_ratio": 0.1}, "workspace_ratio"),
    ({"batching": "static", "mixes_prefill": True}, "static"),
])
def test_profile_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        prof(**kw)
