import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional



# ─────────────────────────────────────────────
# DENSE NEURAL NETWORKS
# ─────────────────────────────────────────────


def get_activation(name: str):
    if name == "SiLU":
        return nn.SiLU
    elif name == "ReLU":
        return nn.ReLU
    elif name == "GELU":
        return nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {name}")


class LBVSDNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = "SiLU",
        input_noise: float = 0.01
    ):
        super().__init__()

        hidden_dims = hidden_dims or [512, 256, 128]

        self.input_noise = input_noise
        act_fn = get_activation(activation)

        layers = []
        curr_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())

            p = dropout if i < len(hidden_dims) - 1 else dropout / 2
            layers.append(nn.Dropout(p))

            curr_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.regressor = nn.Linear(curr_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.input_noise > 0:
            x = x + torch.randn_like(x) * self.input_noise

        x = self.feature_extractor(x)
        return self.regressor(x).squeeze(-1)


def build_dnn(input_dim: int, params: Dict[str, Any]) -> LBVSDNN:
    return LBVSDNN(
        input_dim=input_dim,
        hidden_dims=params.get("hidden_dims", [512, 256, 128]),
        dropout=params.get("dropout", 0.2),
        activation=params.get("activation", "SiLU"),
        input_noise=params.get("input_noise", 0.01)
    )

