import pytest

from servcalc.calibration import PRIORS, Calibration, Curve, get


def test_curve_interpolates_and_clamps():
    c = Curve(((1, 0.5, 0.6, 0.7), (8, 0.6, 0.7, 0.8)))
    assert c.at(1) == (0.5, 0.6, 0.7)
    assert c.at(8) == (0.6, 0.7, 0.8)
    assert abs(c.at(4.5)[1] - 0.65) < 1e-9
    assert c.at(0) == (0.5, 0.6, 0.7)      # clamp below
    assert c.at(100) == (0.6, 0.7, 0.8)    # clamp above


@pytest.mark.parametrize("points,match", [
    (((8, 0.5, 0.6, 0.7), (1, 0.6, 0.7, 0.8)), "ascending"),
    (((1, 0.7, 0.6, 0.5),), "low <= mid <= high"),
    (((1, 0.0, 0.6, 0.7),), "positive"),
    ((), "at least one"),
])
def test_curve_validation(points, match):
    with pytest.raises(ValueError, match=match):
        Curve(points)


def test_priors_exist_and_are_prior():
    for name in ("vllm_bf16", "vllm_int4", "llamacpp_q4", "hf_bf16"):
        cal = get(name)
        assert isinstance(cal, Calibration)
        assert cal.source == "prior"
        lo, mid, hi = cal.overhead_ms
        assert lo <= mid <= hi
    with pytest.raises(KeyError, match="vllm_bf16"):
        get("nope")


def test_vllm_bf16_values():
    cal = PRIORS["vllm_bf16"]
    assert cal.bw_util.at(1) == (0.5, 0.6, 0.7)
    assert cal.mfu.at(512) == (0.3, 0.375, 0.45)
    assert cal.overhead_ms == (1.0, 2.0, 3.0)
