import torch
from torch import nn


class ResumeAnalysisNet(nn.Module):
    def __init__(self, embedding_dim: int, feature_dim: int, hidden_dims=(128, 32), dropout=0.2):
        super().__init__()
        if embedding_dim < 1 or feature_dim < 1 or not hidden_dims or not 0 <= dropout < 1:
            raise ValueError("Invalid network dimensions or dropout")
        self.config = {
            "embedding_dim": embedding_dim,
            "feature_dim": feature_dim,
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
        }
        layers = []
        previous = 4 * embedding_dim + feature_dim
        for size in hidden_dims:
            layers.extend([nn.Linear(previous, size), nn.GELU(), nn.Dropout(dropout)])
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, resume, vacancy, features):
        combined = torch.cat(
            [resume, vacancy, torch.abs(resume - vacancy), resume * vacancy, features], dim=-1
        )
        return self.layers(combined).squeeze(-1)
