import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_td_features(wins: np.ndarray, zc_thresh: float = 0.01, ssc_thresh: float = 0.01) -> np.ndarray:
    n_w, n_c, _ = wins.shape
    feat = np.zeros((n_w, n_c * 5), dtype=np.float32)
    for c in range(n_c):
        sig = wins[:, c, :]
        base = c * 5
        feat[:, base + 0] = np.mean(np.abs(sig), axis=1)
        feat[:, base + 1] = np.sqrt(np.mean(sig ** 2, axis=1))
        feat[:, base + 2] = np.sum(np.abs(np.diff(sig, axis=1)), axis=1)
        sc = np.diff(np.sign(sig), axis=1)
        zc = (np.abs(sc) > 0) & (np.abs(sig[:, :-1]) > zc_thresh)
        feat[:, base + 3] = np.sum(zc, axis=1)
        d = np.diff(sig, axis=1)
        ssc = (np.diff(np.sign(d), axis=1) != 0) & (np.abs(d[:, :-1]) > ssc_thresh)
        feat[:, base + 4] = np.sum(ssc, axis=1)
    return feat


class TinyCNN1D(nn.Module):
    def __init__(self, n_classes: int = 17, n_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyCNN2D(nn.Module):
    def __init__(self, n_classes: int = 17, n_channels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
