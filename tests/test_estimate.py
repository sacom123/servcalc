import json

import pytest

from servcalc import DoesNotFitError, GpuSpec, Workload, estimate, profiles

from .conftest import load_config

L8 = load_config("llama-3.1-8b")


def test_llama8b_3090_vllm_sanity():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512))
    assert e.weight_bytes == 8_030_261_248 * 2
    assert e.weight_stream_bytes == e.weight_bytes - 128_256 * 4096 * 2
    assert e.kv_bytes_per_token == 2 * 32 * 1024 * 2
    assert e.memory_capacity_seqs >= 1
    assert e.configured_capacity_seqs <= 256
    p1 = e.points[0]
    assert p1.batch == 1 and p1.feasible
    assert 15 <= p1.itl_ms.mid <= 40
    assert p1.itl_ms.low <= p1.itl_ms.mid <= p1.itl_ms.high
    assert p1.seq_tok_s.low <= p1.seq_tok_s.mid <= p1.seq_tok_s.high
    assert p1.bound == "memory"
    if e.saturating_batch is not None:
        idx = [p.batch for p in e.points].index(e.saturating_batch)
        assert e.points[idx].bound == "compute" and e.points[idx - 1].bound == "memory"
    assert "steady state" in " ".join(e.assumptions)
    assert e.params.calibration_source == "prior"


def test_ollama_q4_and_num_ctx():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512),
                 profile=profiles.ollama(num_ctx=4096))
    assert e.weight_bytes < 8_030_261_248 * 2 * 0.4
    assert e.configured_capacity_seqs == min(4, e.memory_capacity_seqs)
    assert any("storage" in w for w in e.warnings)
    with pytest.raises(ValueError, match="kv_prealloc_ctx"):
        estimate(L8, "RTX 3090", workload=Workload(1024, 2048), profile="ollama")


def test_hf_static():
    e = estimate(L8, "A100 80GB", workload=Workload(512, 128, concurrency=(1, 8)), profile="hf")
    assert all(p.resident == p.batch for p in e.points)


def test_does_not_fit():
    with pytest.raises(DoesNotFitError):
        estimate(L8, "RTX 3060", workload=Workload(1024, 512))


def test_weight_bytes_override_and_gpuspec():
    g = GpuSpec("mine", 48 * 2**30, 768.0, 155.0)
    e = estimate(L8, g, workload=Workload(1024, 512), profile=profiles.ollama(),
                 weight_bytes=4_900_000_000)
    assert e.weight_bytes == 4_900_000_000


def test_max_users_and_serialization():
    e = estimate(L8, "RTX 3090", workload=Workload(1024, 512))
    assert e.max_users(itl_ms=1e9) == max(p.batch for p in e.points if p.feasible)
    assert e.max_users(itl_ms=0.001) == 0
    d = json.loads(json.dumps(e.to_dict()))
    assert d["points"][0]["itl_ms"]["mid"] == e.points[0].itl_ms.mid
    assert d["params"]["profile"]["name"] == "vllm"
    assert "weights" in e.table() and "tok/s" in e.table()


@pytest.mark.parametrize("name", ["llama-2-7b", "llama-3.2-1b", "llama-3.2-3b",
                                  "mistral-7b-v0.3", "qwen2.5-7b", "qwen2.5-1.5b", "gemma-2-2b"])
def test_all_configs_run(name):
    e = estimate(load_config(name), "A100 80GB",
                 workload=Workload(2048, 256, concurrency=(1, 16)))
    for p in e.points:
        assert p.itl_ms.low <= p.itl_ms.mid <= p.itl_ms.high
        assert p.total_tok_s.low <= p.total_tok_s.mid <= p.total_tok_s.high
