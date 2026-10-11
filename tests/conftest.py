import json
from pathlib import Path

import pytest
from vramcalc import Arch

CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())


@pytest.fixture
def tiny_cfg() -> dict:
    return {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 128, "vocab_size": 256, "tie_word_embeddings": False,
    }


@pytest.fixture
def tiny(tiny_cfg) -> Arch:
    return Arch.from_config(tiny_cfg)
