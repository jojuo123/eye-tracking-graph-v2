"""Small helpers for moving/inspecting batches that may be dicts/lists/tensors."""

import torch


def to_device(batch, device):
    """Recursively move every tensor found in a dict/list/tuple batch to `device`."""
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def infer_batch_size(batch, default=1):
    """Best-effort guess of the batch size by finding the first tensor in the batch."""
    if isinstance(batch, dict):
        for v in batch.values():
            size = infer_batch_size(v, default=None)
            if size is not None:
                return size
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            size = infer_batch_size(v, default=None)
            if size is not None:
                return size
    elif torch.is_tensor(batch):
        return batch.size(0)
    return default
