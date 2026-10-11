import pytest

from servcalc.calibration import Calibration, Curve
from servcalc.gpus import GpuSpec
from servcalc.timing import Range, step_time
from servcalc.work import Work

GPU = GpuSpec("t", 10**9, bandwidth_gbs=1.0, bf16_tflops=0.001)   # 1 GB/s, 1 GFLOP/s
CAL = Calibration(Curve(((1, 0.25, 0.5, 1.0),)), Curve(((1, 0.25, 0.5, 1.0),)),
                  (1.0, 1.0, 1.0), "test")


def test_memory_bound():
    # bytes 1e6 at 1 GB/s: 1 ms at util 1.0, 2 ms at 0.5, 4 ms at 0.25; flops 1e5 -> 0.1 ms
    st = step_time(Work(1e6, 1e5), GPU, CAL, tokens=1)
    assert st.bound == "memory"
    assert st.ms == Range(1.0 + 1, 2.0 + 1, 4.0 + 1)


def test_compute_bound():
    st = step_time(Work(1e5, 1e6), GPU, CAL, tokens=1)
    assert st.bound == "compute"
    assert st.ms.mid == 2.0 + 1


def test_overhead_bound():
    st = step_time(Work(1e3, 1e3), GPU, CAL, tokens=1)
    assert st.bound == "overhead"


def test_range_ops():
    r = Range(1.0, 2.0, 4.0)
    assert r + Range(1.0, 1.0, 1.0) == Range(2.0, 3.0, 5.0)
    assert r.scale(2) == Range(2.0, 4.0, 8.0)
    assert r.invert(1000) == Range(250.0, 500.0, 1000.0)           # low from high
    assert Range.weighted(Range(1, 2, 3), 0.25, Range(5, 6, 7), 0.75) == Range(4.0, 5.0, 6.0)
    with pytest.raises(ValueError, match="low <= mid <= high"):
        Range(3.0, 2.0, 1.0)
