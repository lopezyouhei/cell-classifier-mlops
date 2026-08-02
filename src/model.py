import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_head(cfg: dict, in_dim: int, n_classes: int) -> Head:
    return Head(
        in_dim=in_dim,
        n_classes=n_classes,
        hidden_dim=cfg["head"]["hidden_dim"],
        dropout=cfg["head"]["dropout"],
    )
