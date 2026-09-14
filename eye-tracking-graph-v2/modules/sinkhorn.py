"""Gumbel-Sinkhorn operator for learning latent permutations (Mena, Belanger,
Linderman & Snoek, "Learning Latent Permutations with Gumbel-Sinkhorn
Networks", ICLR 2018), following the reference implementation at
https://github.com/perrying/gumbel-sinkhorn/tree/master.

An unconstrained `(B, N, N)` score matrix ("log_alpha") is pushed towards the
Birkhoff polytope (the set of doubly-stochastic matrices, whose vertices are
exactly the permutation matrices) by alternately normalizing its rows and
columns in log-space -- Sinkhorn iterations. Perturbing `log_alpha` with
Gumbel noise before normalizing turns this into a reparameterizable, continuous
relaxation of *sampling* a random permutation from the distribution `log_alpha`
implies (the "Gumbel-Sinkhorn" trick, analogous to Gumbel-Softmax for
categoricals): as `tau -> 0` and `n_iters -> inf`, the normalized matrix
converges to a hard permutation matrix.
"""

import torch
import torch.nn as nn

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def sample_gumbel(shape, device=None, dtype=None, eps=1e-20):
    """Samples i.i.d. standard Gumbel(0, 1) noise via inverse-CDF sampling."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def sinkhorn_norm(log_alpha, n_iters=20):
    """Projects `(B, N, N)` onto the Birkhoff polytope by alternating row/column
    log-softmax normalization -- the numerically stable, log-domain equivalent
    of repeatedly dividing rows then columns by their sums. Returns the
    doubly-stochastic matrix (probabilities, not logs)."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return log_alpha.exp()


def gumbel_sinkhorn(log_alpha, tau=1.0, n_iters=20, noise_factor=1.0, n_samples=1):
    """Draws `n_samples` continuous relaxations of a random permutation from the
    distribution `log_alpha` implies: perturb with Gumbel noise, sharpen/soften
    with temperature `tau`, then run `sinkhorn_norm`. `log_alpha` is tiled
    `n_samples` times along the batch dimension first, so independent noise is
    used per sample -- averaging the resulting `(B * n_samples, N, N)` matrices'
    loss (grouped every `n_samples` consecutive rows) gives a lower-variance
    training signal than a single sample.
    """
    batch_size, n, _ = log_alpha.shape
    log_alpha = log_alpha.unsqueeze(1).expand(batch_size, n_samples, n, n).reshape(batch_size * n_samples, n, n)
    noise = sample_gumbel(log_alpha.shape, device=log_alpha.device, dtype=log_alpha.dtype)
    log_alpha = (log_alpha + noise * noise_factor) / tau
    return sinkhorn_norm(log_alpha, n_iters)


class GumbelSinkhorn(nn.Module):
    """Wraps `gumbel_sinkhorn`/`sinkhorn_norm` as an `nn.Module` so temperature,
    iteration count, noise scale and sample count live with the rest of a
    model's config.

    Adds Gumbel noise and draws `n_samples` relaxed permutations only in
    training mode. At eval time it returns the deterministic, noise-free
    Sinkhorn normalization of `log_alpha / tau` (a single `(B, N, N)` matrix) --
    pair that with `hungarian_matching(log_alpha)` for a hard, discrete
    prediction.
    """

    def __init__(self, tau=1.0, n_iters=20, noise_factor=1.0, n_samples=1):
        super().__init__()
        self.tau = tau
        self.n_iters = n_iters
        self.noise_factor = noise_factor
        self.n_samples = n_samples

    def forward(self, log_alpha):
        if self.training:
            return gumbel_sinkhorn(log_alpha, self.tau, self.n_iters, self.noise_factor, self.n_samples)
        return sinkhorn_norm(log_alpha / self.tau, self.n_iters)


@torch.no_grad()
def hungarian_matching(scores):
    """Hard, discrete permutation via the Hungarian algorithm (exact linear sum
    assignment), one solve per batch element (scipy has no batched solver).

    `scores`: `(B, N, N)`, where a higher `scores[b, i, j]` means row `i` is a
    better match for column `j` (e.g. `log_alpha`, or a doubly-stochastic
    matrix from `sinkhorn_norm`/`GumbelSinkhorn`).

    Returns a `(B, N)` long tensor `perm` with `perm[b, i]` the column assigned
    to row `i` -- the assignment that *maximizes* the total matched score,
    subject to each row/column being used exactly once.
    """
    if linear_sum_assignment is None:
        raise ImportError("hungarian_matching requires scipy (`pip install scipy`)")
    scores_np = scores.detach().cpu().numpy()
    batch_size, n, _ = scores_np.shape
    perm = torch.empty(batch_size, n, dtype=torch.long)
    for b in range(batch_size):
        _, col_idx = linear_sum_assignment(-scores_np[b])  # negate: solver minimizes cost
        perm[b] = torch.from_numpy(col_idx)
    return perm.to(scores.device)
