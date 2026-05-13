import torch
from torch import nn
from transformers import Trainer, TrainingArguments

from src.training.sft_utils import MidblockMetricsCallback


class TinyMidblockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.midblock = nn.Linear(4, 4)

    def forward(self, input_ids=None, labels=None):
        x = input_ids.float()
        pred = self.midblock(x)
        loss = ((pred - labels.float()) ** 2).mean()
        return {"loss": loss}


def test_midblock_grad_norm_logged_nonzero(tmp_path):
    torch.manual_seed(0)

    model = TinyMidblockModel()
    callback = MidblockMetricsCallback()

    train_dataset = [
        {
            "input_ids": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "labels": torch.tensor([0.5, 1.0, 1.5, 2.0]),
        }
        for _ in range(8)
    ]

    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        max_steps=3,
        logging_steps=1,
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        callbacks=[callback],
    )
    trainer.train()

    assert callback._last_grad_norm > 0.0, (
        f"Expected grad_norm > 0, got {callback._last_grad_norm}. "
        "on_pre_optimizer_step must fire on stored model reference "
        "to capture midblock gradient norm before optimizer step."
    )
