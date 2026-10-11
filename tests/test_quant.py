import pytest

from servcalc import quant


def test_table():
    assert quant.bits(None) == 16
    assert quant.bits("awq") == 4
    assert quant.bits("gguf_q4_k_m") == 4.9
    assert quant.curve_suffix(None) == "bf16"
    assert quant.curve_suffix("gguf_q4_k_m") == "q4"
    assert quant.curve_suffix("awq") == "int4"


def test_unknown():
    with pytest.raises(ValueError, match="bitsandbytes"):
        quant.bits("bitsandbytes")
