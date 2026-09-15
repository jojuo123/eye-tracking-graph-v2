from trainers.losses.losses import LOSSES, build_loss

# Import custom loss modules so their @LOSSES.register(...) decorators run.
from trainers.losses import custom_losses  # noqa: E402,F401
from trainers.losses import permutation_losses  # noqa: E402,F401
from trainers.losses.permutation_losses import permutation_to_matrix

__all__ = ["LOSSES", "build_loss", "permutation_to_matrix"]
