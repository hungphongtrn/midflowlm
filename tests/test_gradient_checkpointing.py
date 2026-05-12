import pytest
import torch

from src.model.sft_flow_midblock import SFTFlowMidblockModel


class _DummyQwen:
    def __init__(self):
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False


def test_gradient_checkpointing_enable_disable():
    """Verify SFTFlowMidblockModel delegates gc to underlying Qwen."""
    model = SFTFlowMidblockModel.__new__(SFTFlowMidblockModel)
    torch.nn.Module.__init__(model)
    model.qwen = _DummyQwen()

    model.gradient_checkpointing_enable()
    assert model.qwen.is_gradient_checkpointing

    model.gradient_checkpointing_disable()
    assert not model.qwen.is_gradient_checkpointing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for VRAM profiling")
def test_gradient_checkpointing_memory_profile():
    model = SFTFlowMidblockModel()
    model = model.cuda().train()

    results = {"gc_off": {}, "gc_on": {}}
    batch_size = 1
    seq_lens = [1024, 1536, 1792, 2048]

    def run_case(seq_len, enable_gc):
        if enable_gc:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
            vocab_size = model.config.text_config.vocab_size
        else:
            raise AttributeError("Unable to resolve vocab_size from model config")

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
        labels = input_ids.clone()

        model.zero_grad(set_to_none=True)
        try:
            outputs = model(input_ids=input_ids, labels=labels)
            outputs.loss.backward()
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            oom = False
        except torch.OutOfMemoryError:
            peak_gb = float("inf")
            oom = True
            torch.cuda.empty_cache()
        fits_12gb = (peak_gb <= 12.0) and not oom
        return peak_gb, fits_12gb

    for seq_len in seq_lens:
        peak_gb, fits_12gb = run_case(seq_len=seq_len, enable_gc=False)
        results["gc_off"][seq_len] = {"peak_gb": peak_gb, "fits_12gb": fits_12gb}

    for seq_len in seq_lens:
        peak_gb, fits_12gb = run_case(seq_len=seq_len, enable_gc=True)
        results["gc_on"][seq_len] = {"peak_gb": peak_gb, "fits_12gb": fits_12gb}

    assert results["gc_on"][1024]["peak_gb"] <= results["gc_off"][1024]["peak_gb"]
