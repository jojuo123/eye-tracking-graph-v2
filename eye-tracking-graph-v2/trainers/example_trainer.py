"""Example trainer: demonstrates the customization point `_optimizer_step` by
adding gradient clipping, controlled via `cfg["trainer"]["grad_clip_norm"]`.

This is the pattern to copy for a new trainer: subclass `BaseTrainer` and
override only the hook(s) you need (`_optimizer_step`, `_train_one_epoch`,
`_validate_and_checkpoint`, ...).
"""

import torch

from trainers.base_trainer import BaseTrainer


class ExampleTrainer(BaseTrainer):
    def _optimizer_step(self):
        clip_norm = self.cfg["trainer"].get("grad_clip_norm")
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        super()._optimizer_step()
