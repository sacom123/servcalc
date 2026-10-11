import dataclasses

import pytest

from servcalc import GpuSpec, Workload, estimate, quant
from servcalc.profiles import BUILTIN, hf, ollama, resolve_profile, vllm


def test_vllm_defaults():
    p = vllm()
    assert (p.batching, p.max_seqs, p.quant, p.kv_block, p.token_budget, p.mixes_prefill) == \
        ("continuous", 256, None, 16, 512, True)
    assert (p.memory_budget_ratio, p.workspace_ratio, p.calibration) == (0.90, 0.05, "vllm_bf16")
    assert p.version == "0.6.3"
    nc = vllm(chunked=False)
    assert nc.token_budget is None and nc.mixes_prefill is False


def test_ollama_defaults_and_args():
    p = ollama()
    assert (p.max_seqs, p.kv_prealloc_ctx, p.quant, p.token_budget, p.mixes_prefill) == \
        (4, 2048, "gguf_q4_k_m", 512, False)
    assert p.calibration == "llamacpp_q4"
    q = ollama(num_parallel=1, num_ctx=8192, quant="gguf_q8_0")
    assert (q.max_seqs, q.kv_prealloc_ctx, q.calibration) == (1, 8192, "llamacpp_q8")


def test_hf_defaults():
    p = hf()
    assert (p.batching, p.max_seqs, p.kv_realloc, p.token_budget, p.workspace_ratio) == \
        ("static", None, True, None, 0.20)
    assert "bf16" in p.version


def test_overrides_and_replace():
    p = vllm(kv_bits=8, max_seqs=32)
    assert (p.kv_bits, p.max_seqs) == (8, 32)
    assert dataclasses.replace(p, quant="awq").weight_bits == 4


def test_resolve_by_name_and_quant_from_config():
    cfg = {"quantization_config": {"quant_method": "awq", "bits": 4}}
    p, warns = resolve_profile("vllm", cfg)
    assert p.quant == "awq" and p.calibration == "vllm_int4"
    assert any("storage" in w for w in warns)
    p2, _ = resolve_profile(vllm(max_seqs=32), cfg)   # unrelated factory args keep config quant
    assert (p2.quant, p2.calibration, p2.max_seqs) == ("awq", "vllm_int4", 32)
    p4, _ = resolve_profile(vllm(quant="fp8"), cfg)   # explicit quant wins over config
    assert p4.quant == "fp8"
    p3, warns3 = resolve_profile("ollama", cfg)       # gguf engines keep their own format, but say so
    assert p3.quant == "gguf_q4_k_m"
    assert any("awq" in w and "gguf_q4_k_m" in w for w in warns3)


def test_gptq_bits_from_config():
    p, _ = resolve_profile("vllm", {"quantization_config": {"quant_method": "gptq", "bits": 8}})
    assert p.quant == "gptq_8" and p.weight_bits == 8
    p4, _ = resolve_profile("vllm", {"quantization_config": {"quant_method": "gptq"}})
    assert p4.quant == "gptq" and p4.weight_bits == 4
    with pytest.raises(ValueError, match="bits"):
        resolve_profile("vllm", {"quantization_config": {"quant_method": "gptq", "bits": 3}})
    with pytest.raises(ValueError, match="bits"):
        resolve_profile("vllm", {"quantization_config": {"quant_method": "awq", "bits": 8}})


@pytest.mark.parametrize("engine", sorted(BUILTIN))
@pytest.mark.parametrize("q", list(quant.QUANTS))
def test_every_engine_quant_pair_estimates(engine, q, tiny_cfg):
    e = estimate(tiny_cfg, GpuSpec("t", 10**9, 100.0, 10.0),
                 workload=Workload(8, 5, concurrency=(1,)), profile=BUILTIN[engine](quant=q))
    assert e.points[0].itl_ms.mid > 0


def test_resolve_rejects_unknown():
    with pytest.raises(KeyError, match="vllm, ollama, hf"):
        resolve_profile("sglang", None)
    with pytest.raises(ValueError, match="bitsandbytes"):
        resolve_profile("vllm", {"quantization_config": {"quant_method": "bitsandbytes"}})
    with pytest.raises(ValueError, match="quant_method"):
        resolve_profile("vllm", {"quantization_config": {"bits": 4}})


def test_builtin_names():
    assert set(BUILTIN) == {"vllm", "ollama", "hf"}
