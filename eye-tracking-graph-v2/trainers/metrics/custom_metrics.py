"""Custom metric examples, registered into the same `METRICS` registry as the
standard ones in `metrics.py`. This is the pattern to copy for your own custom
metric: implement a callable `__call__(outputs, targets) -> scalar tensor` and
decorate with `@METRICS.register("name")`.
"""

import torch

from trainers.metrics.metrics import METRICS


@METRICS.register("topk_accuracy")
class TopKAccuracy:
    """Top-k classification accuracy: correct if the target is among the `k`
    highest-scoring classes (not just the single argmax)."""

    def __init__(self, k=5):
        self.k = k

    @torch.no_grad()
    def __call__(self, outputs, targets):
        k = min(self.k, outputs.size(-1))
        topk = outputs.topk(k, dim=-1).indices
        correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean()


@METRICS.register("dice")
class DiceScore:
    """Dice coefficient (a.k.a. F1 over pixels/voxels) for segmentation masks.

    `outputs`: logits shaped `(N, 1, *spatial)` for binary segmentation, or
    `(N, C, *spatial)` for multiclass. `targets`: `(N, *spatial)` integer
    labels (0/1 for binary). Works for 2D or 3D masks -- `*spatial` can be
    `(H, W)` or `(D, H, W)`.
    """

    def __init__(self, num_classes=1, threshold=0.5, eps=1e-6):
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        dims = tuple(range(1, targets.dim()))
        if self.num_classes == 1:
            preds = (torch.sigmoid(outputs).squeeze(1) > self.threshold).float()
            targets = targets.float()
            intersection = (preds * targets).sum(dim=dims)
            union = preds.sum(dim=dims) + targets.sum(dim=dims)
            return ((2 * intersection + self.eps) / (union + self.eps)).mean()

        preds = outputs.argmax(dim=1)
        scores = []
        for c in range(self.num_classes):
            pred_c, target_c = (preds == c).float(), (targets == c).float()
            intersection = (pred_c * target_c).sum(dim=dims)
            union = pred_c.sum(dim=dims) + target_c.sum(dim=dims)
            scores.append(((2 * intersection + self.eps) / (union + self.eps)).mean())
        return torch.stack(scores).mean()


@METRICS.register("iou")
class IoUScore:
    """Intersection-over-Union for segmentation masks (binary or multiclass,
    2D or 3D) -- see `DiceScore` for the shape conventions."""

    def __init__(self, num_classes=1, threshold=0.5, eps=1e-6):
        self.num_classes = num_classes
        self.threshold = threshold
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        dims = tuple(range(1, targets.dim()))
        if self.num_classes == 1:
            preds = (torch.sigmoid(outputs).squeeze(1) > self.threshold).float()
            targets = targets.float()
            intersection = (preds * targets).sum(dim=dims)
            union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
            return ((intersection + self.eps) / (union + self.eps)).mean()

        preds = outputs.argmax(dim=1)
        scores = []
        for c in range(self.num_classes):
            pred_c, target_c = (preds == c).float(), (targets == c).float()
            intersection = (pred_c * target_c).sum(dim=dims)
            union = pred_c.sum(dim=dims) + target_c.sum(dim=dims) - intersection
            scores.append(((intersection + self.eps) / (union + self.eps)).mean())
        return torch.stack(scores).mean()


@METRICS.register("permutation_accuracy")
class PermutationAccuracy:
    """Fraction of sequence elements assigned to their correct position by a
    hard permutation prediction (e.g. `modules.sinkhorn.hungarian_matching`'s
    output). `outputs`/`targets`: `(B, N)` integer position indices.

    Pass `mask` (`(B, N)` bool, True at valid/non-padded positions, e.g. a batch's
    `fixations_mask`) to exclude padded positions from the average -- without it, a
    padded batch's accuracy would be distorted (e.g. inflated, since
    `modules.sinkhorn.apply_padding_mask` forces padded rows to a trivial self-match)
    by however many padded positions it has, which says nothing about the actual
    prediction.
    """

    @torch.no_grad()
    def __call__(self, outputs, targets, mask=None):
        correct = (outputs == targets).float()
        if mask is None:
            return correct.mean()
        mask = mask.to(correct.dtype)
        return (correct * mask).sum() / mask.sum().clamp_min(1e-8)


@METRICS.register("exact_match")
class ExactMatch:
    """Fraction of samples whose entire *valid* predicted permutation matches the target
    exactly (every non-padded position correct). `outputs`/`targets`: `(B, N)` integer
    position indices.

    Pass `mask` (`(B, N)` bool, True at valid/non-padded positions) so a sample's padded
    tail (whose predicted/target values are otherwise arbitrary) can't spuriously flip its
    exact-match result either way.
    """

    @torch.no_grad()
    def __call__(self, outputs, targets, mask=None):
        correct = outputs == targets
        if mask is not None:
            correct = correct | ~mask  # padded positions are ignored, not required to match
        return correct.all(dim=-1).float().mean()


@METRICS.register("psnr")
class PSNR:
    """Peak Signal-to-Noise Ratio, for image/volume reconstruction quality
    (e.g. autoencoders, super-resolution, diffusion sample quality)."""

    def __init__(self, max_val=1.0, eps=1e-12):
        self.max_val = max_val
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets):
        mse = torch.mean((outputs - targets) ** 2)
        return 10 * torch.log10(self.max_val ** 2 / (mse + self.eps))
