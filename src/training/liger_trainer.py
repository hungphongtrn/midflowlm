from __future__ import annotations

from typing import Any

from transformers import Trainer


class LigerTrainer(Trainer):
    """Trainer that keeps Liger eval loss on the memory-efficient path.

    Liger fused CE skips materializing full ``[batch, seq, vocab]`` logits during
    training by default. Evaluation runs with ``model.eval()``, so force the same
    path when Trainer only needs the prediction loss.
    """

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        if (
            self.args.use_liger_kernel
            and prediction_loss_only
            and "labels" in inputs
            and self._uses_liger_kernel(model)
        ):
            inputs = {**inputs, "skip_logits": True}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _uses_liger_kernel(self, model) -> bool:
        candidates = [model, self.model]
        candidates.extend(getattr(candidate, "module", None) for candidate in list(candidates))

        for candidate in candidates:
            if candidate is None:
                continue
            if getattr(candidate, "uses_liger_kernel", False):
                return True
            qwen = getattr(candidate, "qwen", None)
            forward = getattr(qwen, "forward", None)
            if forward is not None and getattr(forward, "__module__", "").startswith("liger_kernel."):
                return True
        return False
