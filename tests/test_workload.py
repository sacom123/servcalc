import pytest

from servcalc.workload import Workload


def test_defaults():
    w = Workload(prompt_len=1024, gen_len=512)
    assert w.concurrency == (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert w.decode_steps == 511
    assert w.mean_decode_ctx == 1024 + 1 + 510 / 2


@pytest.mark.parametrize("kwargs,match", [
    ({"prompt_len": 0, "gen_len": 2}, "prompt_len"),
    ({"prompt_len": 8, "gen_len": 1}, "gen_len"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": (0, 1)}, "concurrency"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": (4, 2)}, "ascending"),
    ({"prompt_len": 8, "gen_len": 2, "concurrency": ()}, "concurrency"),
])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Workload(**kwargs)
