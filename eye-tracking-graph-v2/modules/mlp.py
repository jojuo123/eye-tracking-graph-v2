"""A simple configurable multi-layer perceptron building block."""

import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}


class MLP(nn.Module):
    """Stack of Linear -> (Norm) -> Activation -> (Dropout), ending in a final Linear
    projection to `out_dim` (no activation on the output)."""

    def __init__(self, in_dim, hidden_dims, out_dim, activation="relu", dropout=0.0, norm=False):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', choose from {list(_ACTIVATIONS)}")
        act_layer = _ACTIVATIONS[activation]

        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
