"""Inferrer: runs a model over unlabeled inputs and returns raw predictions."""

import torch

from utils.tensor import to_device


class Inferrer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, batch):
        """Run inference on a single batch, returning `output["outputs"]` (or the
        full output dict if the model didn't include an "outputs" key)."""
        was_training = self.model.training
        self.model.eval()
        batch = to_device(batch, self.device)
        output = self.model(batch)
        if was_training:
            self.model.train()
        return output.get("outputs", output)

    @torch.no_grad()
    def predict_dataloader(self, dataloader):
        """Run inference over an entire dataloader, concatenating tensor outputs."""
        outputs = [self.predict(batch) for batch in dataloader]
        if outputs and all(torch.is_tensor(o) for o in outputs):
            return torch.cat(outputs, dim=0)
        return outputs
