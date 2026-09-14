# PyTorch Deep Learning Template

A minimal, extensible project layout for PyTorch projects, organized around a
registry pattern so new models/datasets/losses/optimizers/schedulers/trainers
can be added by writing one file and adding one decorator — no other file
needs to change.

## Structure

```
pytorch/
├── README.md
├── requirements.txt
├── mount_erda.sh / unmount_erda.sh    # mount/unmount the external data share
└── eye-tracking-graph-v2/    # all code lives here
    ├── main.py                 # entry point: load config -> build trainer -> train
    ├── configs/
    │   └── default.yaml        # example config, runs out of the box on synthetic data
    ├── models/                    # model architectures
    │   ├── base_model.py          #   BaseModel contract (see below)
    │   ├── example_model.py       #   MLPClassifier, registered as "mlp_classifier"
    │   └── sinkhorn_sort_model.py  #   SinkhornPatchSorter, registered as "sinkhorn_patch_sorter"
    ├── modules/                   # reusable nn.Module building blocks
    │   ├── mlp.py                 #   MLP
    │   ├── cnn.py                 #   ConvBlock, SimpleCNNEncoder
    │   ├── transformer.py         #   PositionalEncoding, TransformerEncoderBlock/Encoder
    │   ├── nd.py                  #   get_nd_layers/match_and_concat: shared 2D/3D plumbing
    │   ├── resnet.py               #   BasicBlock, ResNetEncoder            (2D/3D)
    │   ├── unet.py                 #   DoubleConv, UNetEncoder/Decoder, UNet (2D/3D)
    │   ├── resnet_unet.py           #   ResNetUNet: ResNet encoder + U-Net decoder (2D/3D)
    │   ├── diffusion.py             #   SinusoidalTimeEmbedding, DiffusionUNet, GaussianDiffusion (2D/3D)
    │   ├── vision_transformer.py    #   PatchEmbedding, VisionTransformer     (2D/3D)
    │   ├── sinkhorn.py               #   GumbelSinkhorn, sinkhorn_norm, gumbel_sinkhorn, hungarian_matching
    │   └── encoders.py                #   ENCODERS registry: swappable image encoders (simple_cnn/resnet/vit)
    ├── dataloaders/                # datasets + dataloader construction
    │   ├── base_dataset.py         #   BaseDataset contract
    │   ├── example_dataset.py      #   SyntheticClassificationDataset
    │   └── patch_sequence_dataset.py#   SyntheticPatchSequenceDataset, registered as "synthetic_patch_sequence"
    ├── trainers/                   # training/eval/inference orchestration
    │   ├── base_trainer.py         #   BaseTrainer: build(), train(), resume()
    │   ├── example_trainer.py      #   subclass demonstrating a hook override
    │   ├── evaluator.py            #   Evaluator: aggregates val losses/metrics
    │   ├── inferrer.py             #   Inferrer: raw prediction / inference
    │   ├── losses/losses.py        #   LOSSES registry: built-in wrappers (nn.CrossEntropyLoss, ...)
    │   ├── losses/custom_losses.py #   Custom losses: FocalLoss, WeightedMultiLoss
    │   ├── metrics/metrics.py       #   METRICS registry: accuracy, precision, recall, f1, mae, mse
    │   ├── metrics/custom_metrics.py#   Custom metrics: TopKAccuracy, DiceScore, IoUScore, PSNR, PermutationAccuracy, ExactMatch
    │   ├── metrics/collection.py    #   MetricCollection: evaluate several metrics together
    │   ├── optimizers/optimizers.py#   OPTIMIZERS registry (Adam, AdamW, SGD, ...)
    │   └── schedulers/schedulers.py#   SCHEDULERS registry (Cosine, StepLR, ...)
    ├── utils/                       # cross-cutting helpers
    │   ├── registry.py              #   Registry: name -> class, `.build(cfg)`
    │   ├── config.py                #   YAML loading + attribute-style dict access
    │   ├── checkpoint.py             #   CheckpointManager: save/prune/best/resume
    │   ├── logger.py                  #   get_logger (console + file)
    │   ├── meter.py                   #   AverageMeter / MetricTracker
    │   ├── tensor.py                  #   to_device / infer_batch_size
    │   ├── seed.py                    #   set_seed
    │   ├── ema.py                     #   ModelEMA (example "algorithm" helper)
    │   └── h5.py                      #   write_h5/write_dataframe: HDF5 writing helpers
    └── data/                        # dataset-specific preprocessing scripts
        └── reflacx/reflacx.py        #   REFLACX metadata/fixations/image -> H5 preprocessing
```

## Quick start

```bash
pip install -r requirements.txt
cd eye-tracking-graph-v2
python main.py --config configs/default.yaml
```

This trains `MLPClassifier` on a synthetic clustered-classification dataset
for 10 epochs, logging to stdout and `work_dir/example_run/train.log`, and
saving checkpoints (plus a running `checkpoint_best.pt`) under
`work_dir/example_run/checkpoints/`.

All other commands below (and the `configs/`, `models/`, etc. paths they
reference) assume you're still inside `eye-tracking-graph-v2/` from the `cd`
above.

Resume training:

```bash
python main.py --config configs/default.yaml --resume work_dir/example_run/checkpoints/checkpoint_best.pt
```

## The model contract

Every model subclasses `models.base_model.BaseModel` and implements:

```python
def forward(self, batch):
    ...
    return {
        "outputs": logits,                          # always present
        "loss": loss,                                # scalar tensor, only if targets given
        "losses": {"cls_loss": loss.detach()},        # for logging
        "metrics": {"accuracy": accuracy.detach()},   # for logging
    }
```

Keeping loss/metric computation *inside* the model (rather than the trainer)
means `BaseTrainer`'s training loop is generic across completely different
tasks (classification, regression, seq2seq, ...) — it just reads
`output["loss"]`, `output["losses"]`, `output["metrics"]`.

## Adding a new model

1. Create `models/my_model.py`, subclass `BaseModel`, build it from `modules/`.
2. Register it:
   ```python
   from models import MODELS

   @MODELS.register("my_model")
   class MyModel(BaseModel):
       ...
   ```
3. Import the file once from `models/__init__.py` (so the decorator runs).
4. Reference it from a config: `model: {type: my_model, ...}`.

The same pattern applies to datasets (`dataloaders/`, `DATASETS` registry),
losses (`trainers/losses`, `LOSSES`), metrics (`trainers/metrics`, `METRICS`),
optimizers (`trainers/optimizers`, `OPTIMIZERS`), schedulers
(`trainers/schedulers`, `SCHEDULERS`), and trainers (`trainers/`, `TRAINERS`).

## Custom losses

`trainers/losses/losses.py` registers thin wrappers around built-in
`torch.nn` losses (`cross_entropy`, `mse`, `l1`, ...). `trainers/losses/custom_losses.py`
shows the pattern for writing your own:

```python
from trainers.losses.losses import LOSSES

@LOSSES.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        ...
    def forward(self, logits, targets):
        ...
```

Two examples are included:
- **`FocalLoss`** (`type: focal`) — down-weights easy/well-classified examples
  via `(1 - p_t) ** gamma`, useful for class-imbalanced classification.
  `configs/focal_loss_example.yaml` is `configs/default.yaml` with the loss
  swapped in through config alone:
  ```bash
  python main.py --config configs/focal_loss_example.yaml
  ```
  This works because `MLPClassifier` (`models/example_model.py`) builds its
  loss via `build_loss(loss_cfg or {"type": "cross_entropy"})` instead of
  hardcoding `F.cross_entropy` — any registered loss can be dropped in
  through `model.loss_cfg` without touching model code.
- **`WeightedMultiLoss`** (`type: weighted_multi`) — combines several
  registered losses into one weighted total for multi-term objectives
  (e.g. blending `cross_entropy` with an auxiliary loss, or a VAE's
  `recon_loss + beta * kl_loss`). It returns a breakdown dict so each term
  can be logged separately:
  ```python
  loss_fn = build_loss({
      "type": "weighted_multi",
      "terms": {
          "ce": {"type": "cross_entropy", "weight": 1.0},
          "focal": {"type": "focal", "weight": 0.5, "gamma": 2.0},
      },
  })
  result = loss_fn(logits, targets)   # {"ce": ..., "focal": ..., "loss": ...}
  ```

## Custom metrics

`trainers/metrics/metrics.py` registers standard metrics (`accuracy`,
`precision`, `recall`, `f1`, `mae`, `mse`) as small callables:
`metric(outputs, targets) -> scalar tensor`. `trainers/metrics/custom_metrics.py`
shows the pattern for writing your own:

```python
from trainers.metrics.metrics import METRICS

@METRICS.register("topk_accuracy")
class TopKAccuracy:
    def __init__(self, k=5):
        ...
    def __call__(self, outputs, targets):
        ...
```

Three examples are included:
- **`TopKAccuracy`** (`type: topk_accuracy`) — correct if the target is among
  the `k` highest-scoring classes, not just the argmax.
- **`DiceScore`** / **`IoUScore`** (`type: dice` / `type: iou`) — overlap
  metrics for segmentation masks, binary or multiclass, 2D or 3D.
- **`PSNR`** (`type: psnr`) — reconstruction quality for autoencoders,
  super-resolution, or diffusion sample quality.

Like losses, a model builds a configurable set of metrics via `metrics_cfg`
instead of hardcoding them. `MLPClassifier` (`models/example_model.py`) does:

```python
self.metrics = MetricCollection(metrics_cfg or [{"type": "accuracy"}])
...
def compute_metrics(self, outputs, targets):
    return self.metrics(outputs, targets)   # {"accuracy": ..., "precision": ..., ...}
```

`configs/custom_metrics_example.yaml` swaps in precision/recall/f1/top-k
accuracy through config alone (also showing the `name` alias, for reusing
`topk_accuracy` twice with different `k`):

```bash
python main.py --config configs/custom_metrics_example.yaml
```

Use `MetricCollection` directly for a one-off evaluation, too:

```python
from trainers.metrics import MetricCollection

metrics = MetricCollection([
    {"type": "dice", "num_classes": 1},
    {"type": "iou", "num_classes": 1},
])
print(metrics(pred_masks, target_masks))   # {"dice": 0.91, "iou": 0.84}
```

## Modules: 2D/3D building blocks

Beyond `MLP`/`ConvBlock`/`TransformerEncoder`, `modules/` includes common
architectures. Each takes a `spatial_dims` argument (`2` for images, `3` for
volumes/video) via the shared `modules/nd.py` helper (`get_nd_layers` swaps
`Conv2d`/`BatchNorm2d`/... for their 3D counterparts; `match_and_concat` pads
and joins encoder/decoder skip connections):

| Module | File | Notes |
|---|---|---|
| `ResNetEncoder` | `resnet.py` | Stem + N `BasicBlock` stages; `return_intermediate=True` gives multi-scale features |
| `UNet` / `UNetEncoder` / `UNetDecoder` | `unet.py` | Classic encoder-decoder with skip connections |
| `ResNetUNet` | `resnet_unet.py` | `ResNetEncoder` backbone + U-Net-style decoder |
| `DiffusionUNet` / `GaussianDiffusion` | `diffusion.py` | Time-conditioned U-Net denoiser + DDPM forward process/training loss |
| `VisionTransformer` / `PatchEmbedding` | `vision_transformer.py` | Patch embed (Conv2d/Conv3d) + `[CLS]` token + `TransformerEncoderBlock` stack |

All support 2D and 3D, verified with real forward/backward passes:

```python
from modules import UNet, ResNetUNet, VisionTransformer, DiffusionUNet, GaussianDiffusion

# 2D image segmentation
unet2d = UNet(in_channels=3, out_channels=1, channels=(64, 128, 256, 512), spatial_dims=2)
mask = unet2d(torch.randn(4, 3, 256, 256))          # (4, 1, 256, 256)

# 3D volumetric segmentation (e.g. medical imaging)
unet3d = UNet(in_channels=1, out_channels=3, channels=(16, 32, 64), spatial_dims=3)
mask = unet3d(torch.randn(2, 1, 64, 64, 64))         # (2, 3, 64, 64, 64)

# ResNet-backbone U-Net
model = ResNetUNet(in_channels=3, out_channels=1, spatial_dims=2)

# Vision Transformer, 2D image or 3D volume/video
vit2d = VisionTransformer(image_size=224, patch_size=16, in_channels=3, num_classes=1000, spatial_dims=2)
vit3d = VisionTransformer(image_size=32, patch_size=8, in_channels=1, num_classes=10, spatial_dims=3)

# Diffusion: predict noise, then compute the DDPM training loss
denoiser = DiffusionUNet(in_channels=3, out_channels=3, spatial_dims=2)
diffusion = GaussianDiffusion(num_timesteps=1000)
loss = diffusion.training_loss(denoiser, x0)   # x0: clean images/volumes
```

These are building blocks, not registered `MODELS` — wrap one in a
`BaseModel` subclass (see `models/example_model.py`) to plug it into a
trainer, e.g. a `UNetSegmentation` model returning
`{"outputs": logits, "loss": dice_loss, "metrics": {"dice": ..., "iou": ...}}`.

## Sequence reordering with Gumbel-Sinkhorn networks

`modules/sinkhorn.py` implements the Gumbel-Sinkhorn operator (Mena et al.,
2018, "Learning Latent Permutations with Gumbel-Sinkhorn Networks"), following
[perrying/gumbel-sinkhorn](https://github.com/perrying/gumbel-sinkhorn/tree/master):
an unconstrained `(B, N, N)` score matrix is pushed onto the Birkhoff polytope
(doubly-stochastic matrices, whose vertices are permutation matrices) by
alternating row/column log-softmax normalization ("Sinkhorn iterations").
Perturbing the score matrix with Gumbel noise first makes this a
reparameterizable, continuous relaxation of *sampling* a random permutation —
the permutation analogue of Gumbel-Softmax for categoricals.

```python
from modules.sinkhorn import GumbelSinkhorn, hungarian_matching

sinkhorn = GumbelSinkhorn(tau=0.5, n_iters=20, noise_factor=1.0, n_samples=5)
soft_perm = sinkhorn(log_alpha)              # (B, N, N) doubly-stochastic, training only adds noise
hard_perm = hungarian_matching(log_alpha)    # (B, N) exact permutation via linear-sum assignment
```

`models/sinkhorn_sort_model.py` builds `SinkhornPatchSorter` (`type:
sinkhorn_patch_sorter`) on top of it: given a sequence of `N` `(2D coordinate,
image patch)` pairs in some shuffled order, it predicts the permutation that
recovers their correct order (jigsaw-style shuffled patches, or eye-tracking
fixations — a patch plus its coordinate — in scrambled temporal order both fit
this shape). Each patch is encoded with a swappable backbone, each coordinate
with an `MLP`, summed into one token per element, passed through a
`TransformerEncoder`, then scored pairwise into an `(N, N)` `log_alpha` matrix.
`GumbelSinkhorn` turns that into a soft permutation matrix for a Sinkhorn/NLL
training loss against the ground-truth permutation; `hungarian_matching` turns
it into the hard, discrete prediction used for metrics/inference.

The patch backbone is picked via `model.encoder_cfg` through `modules/encoders.py`'s
`ENCODERS` registry (mirroring `LOSSES`/`METRICS`), so it's a config change, not
a model change:

| `encoder_cfg.type` | Backbone | Notes |
|---|---|---|
| `simple_cnn` (default) | `SimpleCNNEncoder` | cheapest, good for small patches |
| `resnet` | `ResNetEncoder` (`modules/resnet.py`) | more capacity; `layers`/`base_channels` configurable |
| `vit` | `PatchEmbedding` + `TransformerEncoder` | ViT-style, no `[CLS]`/head, mean-pooled |

`in_channels` and `out_dim` are always forced onto whichever encoder is built
(`out_dim` to `embed_dim`, so patch and coordinate features can be summed);
every other key in `encoder_cfg` is forwarded to the chosen encoder's
constructor. Add a custom backbone by subclassing `nn.Module` with
`forward(x) -> (B, out_dim)` and `@ENCODERS.register("name")`.

`configs/sinkhorn_sort_example.yaml` (`simple_cnn`) and
`configs/sinkhorn_sort_resnet_example.yaml` (`resnet`) run the same pipeline
against `dataloaders/patch_sequence_dataset.py`'s `SyntheticPatchSequenceDataset`
(a random-walk of coordinates plus intensity-coded patches, shuffled by a known
permutation — recovering it is the task), logging `permutation_accuracy`
(fraction of elements correctly placed) and `exact_match`
(`trainers/metrics/custom_metrics.py`) alongside the loss:

```bash
python main.py --config configs/sinkhorn_sort_example.yaml
python main.py --config configs/sinkhorn_sort_resnet_example.yaml
```

## Adding a new trainer

Most customizations are a small override on top of `BaseTrainer`. See
`trainers/example_trainer.py`, which adds gradient clipping by overriding the
`_optimizer_step` hook. Other hooks: `_train_one_epoch`,
`_step_scheduler_post_epoch`, `_validate_and_checkpoint`.

`BaseTrainer` itself is built with a single `build()` method:

```python
trainer = build_trainer(cfg)   # picks the class via cfg["trainer"]["type"]
trainer.build()                # constructs model/data/optimizer/scheduler/evaluator/inferrer
trainer.train()                # runs the training loop
```

After `build()`, `trainer.evaluator` (an `Evaluator`) and `trainer.inferrer`
(an `Inferrer`) are ready to use directly, e.g. for a standalone eval/predict
script:

```python
from trainers import build_trainer
from utils.config import load_config

cfg = load_config("configs/default.yaml")
trainer = build_trainer(cfg).build()
trainer.resume("work_dir/example_run/checkpoints/checkpoint_best.pt")

print(trainer.evaluator.evaluate())          # {"accuracy": 0.87, ...}
preds = trainer.inferrer.predict(some_batch) # raw model outputs
```
