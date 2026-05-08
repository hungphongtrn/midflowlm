import pytest
import torch
from src.model.sft_flow_midblock import SFTFlowMidblockModel

CHECKPOINT_PATH = "models/p3_d3_mix_c/checkpoint.pth"


@pytest.fixture(scope="module")
def model():
    return SFTFlowMidblockModel(checkpoint_path=CHECKPOINT_PATH)


class TestSFTFlowMidblockParameterCounts:
    def test_only_midblock_is_trainable(self, model):
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
        for name in trainable_names:
            assert "midblock" in name, f"Trainable param '{name}' not in midblock"
        for name in frozen_names:
            assert "midblock" not in name, f"Frozen param '{name}' is in midblock"
        assert model.trainable_params > 0
        assert model.frozen_params > 0

    def test_midblock_param_count_matches_checkpoint(self, model):
        assert 19_000_000 <= model.trainable_params <= 25_000_000, \
            f"Expected ~20M-25M trainable params, got {model.trainable_params:,}"

    def test_total_params_under_one_billion(self, model):
        total = model.trainable_params + model.frozen_params
        assert total < 1_000_000_000, f"Total params {total:,} exceeds 1B"


class TestSFTFlowMidblockWarmStart:
    def test_midblock_weights_match_checkpoint(self):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        ckpt_state = checkpoint["model_state_dict"]
        model = SFTFlowMidblockModel(checkpoint_path=CHECKPOINT_PATH)
        key = "velocity_proj.1.weight"
        checkpoint_weight = ckpt_state[f"midblock.{key}"]
        model_weight = model.midblock.velocity_proj[1].weight.data
        assert torch.allclose(checkpoint_weight, model_weight, atol=1e-6), f"Weight mismatch for {key}"


class TestSFTFlowMidblockForward:
    def test_forward_produces_logits(self, model):
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones(2, 64)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert "logits" in outputs
        assert outputs["logits"].shape[:2] == (2, 64)
        assert outputs["logits"].shape[2] > 100_000  # vocab size for 0.8B is 248320

    def test_forward_with_labels_produces_loss(self, model):
        model.train()
        input_ids = torch.randint(0, 1000, (2, 64))
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].requires_grad
        assert outputs["loss"].item() > 0

    def test_gradients_only_flow_to_midblock(self, model):
        model.train()
        input_ids = torch.randint(0, 1000, (2, 64))
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        for name, p in model.named_parameters():
            if "midblock" in name:
                assert p.grad is not None, f"Midblock param '{name}' has no gradient"
            else:
                assert p.grad is None, f"Frozen param '{name}' has gradient"
