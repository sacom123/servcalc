import json
import subprocess
import sys
from pathlib import Path

from .conftest import CONFIG_DIR

CFG = str(CONFIG_DIR / "llama-3.1-8b.json")


def run(*args):
    return subprocess.run([sys.executable, "-m", "servcalc.cli", *args],
                          capture_output=True, text=True, check=False)


def test_table_output():
    r = run(CFG, "--gpu", "RTX 3090")
    assert r.returncode == 0, r.stderr
    assert "capacity:" in r.stdout and "tok/s" in r.stdout


def test_json_and_slo():
    r = run(CFG, "--gpu", "A100 80GB", "--engine", "vllm", "--prompt", "2048", "--gen", "256",
            "--concurrency", "1,8,64", "--slo-itl", "50", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert [p["batch"] for p in d["points"]] == [1, 8, 64]
    assert "max_users" in d and isinstance(d["max_users"], int)


def test_ollama_options():
    r = run(CFG, "--gpu", "3090", "--engine", "ollama", "--max-seqs", "2", "--num-ctx", "4096",
            "--quant", "gguf_q8_0", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["params"]["profile"]["max_seqs"] == 2
    assert d["params"]["profile"]["quant"] == "gguf_q8_0"


def test_does_not_fit_exit_1():
    r = run(CFG, "--gpu", "RTX 3060")
    assert r.returncode == 1
    assert "exceed" in r.stderr and "Traceback" not in r.stderr


def test_bad_args_exit_2():
    assert run(CFG, "--gpu", "24").returncode == 2
    assert run(CFG, "--gpu", "RTX 3090", "--gen", "1").returncode == 2
    assert run(CFG, "--gpu", "RTX 3090", "--engine", "sglang").returncode == 2


def test_gpus_subcommand():
    r = run("gpus")
    assert r.returncode == 0 and "RTX A6000" in r.stdout and "GB/s" in r.stdout


def test_flags_keep_config_quant(tmp_path):
    cfg = json.loads(Path(CFG).read_text())
    cfg["quantization_config"] = {"quant_method": "awq", "bits": 4}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    r = run(str(path), "--gpu", "RTX 3090", "--max-seqs", "32", "--kv-bits", "8", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    prof = d["params"]["profile"]
    assert (prof["quant"], prof["max_seqs"], prof["kv_bits"], prof["calibration"]) == \
        ("awq", 32, 8, "vllm_int4")
    assert d["weight_bytes"] < 8_030_261_248 * 2 * 0.4


def test_engine_specific_flags_rejected():
    r = run(CFG, "--gpu", "RTX 3090", "--num-ctx", "4096")
    assert r.returncode == 2 and "ollama" in r.stderr
    r = run(CFG, "--gpu", "RTX 3090", "--engine", "hf", "--no-chunked")
    assert r.returncode == 2 and "vllm" in r.stderr


def test_numeric_gpu_hint():
    r = run(CFG, "--gpu", "24")
    assert r.returncode == 2 and "servcalc gpus" in r.stderr
