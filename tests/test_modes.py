import math

import pytest

from servcalc.calibration import Calibration, Curve
from servcalc.gpus import GpuSpec
from servcalc.modes import DoesNotFitError, capacity, point
from servcalc.profiles import EngineProfile
from servcalc.work import Weights
from servcalc.workload import Workload

GPU = GpuSpec("t", 1_000_000, 1.0, 0.001)
CAL = Calibration(Curve(((1, 0.5, 0.5, 0.5),)), Curve(((1, 0.5, 0.5, 0.5),)),
                  (1.0, 1.0, 1.0), "test")
WTS = Weights(213_632, 180_864)
W = Workload(prompt_len=8, gen_len=5, concurrency=(1, 2, 4))


def prof(**kw) -> EngineProfile:
    base = {"name": "t", "version": "0", "batching": "continuous", "max_seqs": None, "quant": None,
                "kv_bits": 16, "kv_block": 0, "kv_prealloc_ctx": None, "memory_budget_ratio": 1.0,
                "workspace_ratio": 0.1, "token_budget": 6, "mixes_prefill": True, "kv_realloc": False,
                "calibration": "vllm_bf16"}
    return EngineProfile(**{**base, **kw})


def ms(bytes_, flops):
    return max(bytes_ / (1e9 * 0.5) * 1000, flops / (1e9 * 0.5) * 1000) + 1.0


def test_continuous_mixed_values(tiny):
    p = prof()
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert b.resident == 3                      # M = ceil(0.5 * 2) = 1
    assert b.feasible and b.infeasible_reason is None
    # chunk steps: s_mix(4, 0) = 187776 B / 969984 F, s_mix(4, 4, last) = 188800 B / 1010944 F
    ttft = ms(187_776, 969_984) + ms(188_800, 1_010_944)
    assert math.isclose(b.ttft_ms.mid, ttft)
    assert math.isclose(b.itl_ms.mid, 1.0 * ttft / 2 + 0.0 * ms(186_752, 372_480))   # f = 1
    assert math.isclose(b.total_tok_s.mid, 2 * 1000 / b.itl_ms.mid)
    assert b.bound == "compute"


def test_continuous_f_weighting_tends_to_decode_only(tiny):
    # large B makes f = lambda*I/c small
    p = prof(token_budget=10_000)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    f = 0.5 * 8 / (10_000 - 2)
    assert 0 < f < 0.01
    assert b.itl_ms.mid < ms(186_752, 372_480) * 1.05


def test_continuous_budget_infeasible(tiny):
    p = prof(token_budget=6)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    # N=4: lambda = 1, N + lambda*I = 12 > 6
    b = point(4, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert not b.feasible and b.infeasible_reason == "token_budget"
    assert b.ttft_ms is not None                 # c = 2 >= 1 so prefill is still representable
    # N=6: c = 0
    w6 = Workload(8, 5, concurrency=(6,))
    cap6 = capacity(GPU, WTS.resident, 256, w6, p)
    b6 = point(6, tiny, GPU, w6, p, CAL, WTS, 256, cap6)
    assert b6.ttft_ms is None and b6.infeasible_reason == "token_budget"
    assert math.isclose(b6.itl_ms.mid, ms(180_864 + 256 * (6 * 10.5 + 6),
                                          2 * 90_432 * 6 + 4 * 2 * 64 * 6 * 10.5))


def test_continuous_budget_none_means_whole_prefill(tiny):
    p = prof(token_budget=None)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(4, tiny, GPU, W, p, CAL, WTS, 256, cap)     # lambda = 1 -> f = min(1, 1) = 1
    assert b.feasible
    assert math.isclose(b.ttft_ms.mid, ms(180_864 + 256 * (4 * 10.5 + 4) + 256 * 8,
                                         2 * 90_432 * 4 + 4 * 2 * 64 * 4 * 10.5 + 1_235_968))


def test_continuous_unmixed(tiny):
    p = prof(mixes_prefill=False, token_budget=6)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    s_d = ms(186_752, 372_480)
    # chunks (6,0,False): bytes 180864 - 32768 + 256*6 = 149632, flops 899328
    #        (2,6,True):  bytes 180864 + 256*8 = 182912, flops 336640
    t_p = ms(149_632, 899_328) + ms(182_912, 336_640)
    assert math.isclose(b.itl_ms.mid, s_d + 0.5 * t_p)
    assert math.isclose(b.ttft_ms.mid, t_p + 0.5 * s_d)
    assert b.resident == 2 + math.ceil(0.5 * t_p / (s_d + 0.5 * t_p))


def test_static(tiny):
    p = prof(batching="static", mixes_prefill=False, token_budget=None)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)
    assert b.resident == 2
    assert math.isclose(b.ttft_ms.mid, ms(184_960, 2_471_936))           # N*I prefill
    steps = [ms(180_864 + 256 * (2 * (9 + j) + 2), 2 * 90_432 * 2 + 4 * 2 * 64 * 2 * (9 + j))
             for j in range(4)]
    assert math.isclose(b.itl_ms.mid, sum(steps) / 4)
    assert steps == sorted(steps)                                          # monotone in j


def test_max_seqs_and_memory_reasons(tiny):
    p = prof(max_seqs=2)
    cap = capacity(GPU, WTS.resident, 256, W, p)
    b = point(2, tiny, GPU, W, p, CAL, WTS, 256, cap)                     # R = 3 > 2
    assert b.infeasible_reason == "max_seqs"
    small = GpuSpec("s", 220_000, 1.0, 0.001)             # usable = 220000 - 213632 - 22000 < 0
    with pytest.raises(DoesNotFitError):
        capacity(small, WTS.resident, 256, W, p)


def test_continuous_unmixed_tied_skips_lm_head_on_non_last_chunks(tiny_cfg):
    from vramcalc import Arch

    from servcalc import work
    tied = Arch.from_config({**tiny_cfg, "tie_word_embeddings": True})
    wts = work.weight_bytes(tied, 16)              # stream == resident == 213632 - 32768
    fast = GpuSpec("fast", 1_000_000, 1.0, 1.0)    # 1 TFLOP/s: every step is memory-bound
    p = prof(mixes_prefill=False, token_budget=6)
    cap = capacity(fast, wts.resident, 256, W, p)
    b = point(2, tied, fast, W, p, CAL, wts, 256, cap)

    def mem_ms(bytes_):
        return bytes_ / (1e9 * 0.5) * 1000 + 1.0

    s_d = mem_ms(wts.stream + 256 * (2 * 10.5 + 2))
    first = mem_ms(wts.stream - 32_768 + 256 * 6)   # no lm_head on a non-last chunk, tied or not
    last = mem_ms(wts.stream + 256 * 8)
    assert math.isclose(b.itl_ms.mid, s_d + 0.5 * (first + last))
