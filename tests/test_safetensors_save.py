from collections import defaultdict

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.model.sft_flow_midblock import SFTFlowMidblockModel


@pytest.fixture
def tiny_sft_model(monkeypatch):
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return config

    def fake_auto_model_from_pretrained(*args, **kwargs):
        return Qwen2ForCausalLM(config)

    monkeypatch.setattr("src.model.sft_flow_midblock.AutoConfig.from_pretrained", fake_auto_config_from_pretrained)
    monkeypatch.setattr("src.model.sft_flow_midblock.AutoModelForCausalLM.from_pretrained", fake_auto_model_from_pretrained)

    return SFTFlowMidblockModel(
        model_name="dummy-qwen2",
        start_layer=0,
        end_layer=1,
        thinking_level=2,
        checkpoint_path=None,
        torch_dtype=torch.float32,
    )


def _shared_key_groups(state_dict):
    grouped = defaultdict(list)
    for key, tensor in state_dict.items():
        grouped[id(tensor)].append(key)
    return [keys for keys in grouped.values() if len(keys) > 1]


def test_state_dict_no_shared_tensors(tiny_sft_model):
    state = tiny_sft_model.state_dict()
    shared_groups = _shared_key_groups(state)
    assert not shared_groups, (
        f"Expected zero shared tensor references after dedup, "
        f"found: {shared_groups}"
    )


def test_state_dict_round_trip_via_torch_save(tmp_path, tiny_sft_model):
    save_path = tmp_path / "model_state.pth"
    state = tiny_sft_model.state_dict()
    torch.save(state, save_path)
    loaded = torch.load(save_path, map_location="cpu", weights_only=True)
    for k, v in state.items():
        assert torch.equal(v, loaded[k]), f"Mismatch for key {k}"

    # And load back into a fresh model
    model2 = type(tiny_sft_model).__new__(type(tiny_sft_model))
    model2_state = tiny_sft_model.state_dict()
    torch.save(model2_state, tmp_path / "model2.pth")
    loaded2 = torch.load(tmp_path / "model2.pth", map_location="cpu", weights_only=True)
    assert loaded2.keys() == model2_state.keys()


def test_torch_save_state_dict_still_works(tmp_path, tiny_sft_model):
    save_path = tmp_path / "model_state.pth"
    torch.save(tiny_sft_model.state_dict(), save_path)
    loaded = torch.load(save_path, map_location="cpu", weights_only=True)

    assert save_path.exists()
    assert loaded.keys() == tiny_sft_model.state_dict().keys()
