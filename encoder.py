import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError


class ConvEncoder(Encoder):
    def __init__(self, dim=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(self.body(x).squeeze(-1))

    @property
    def dim(self):
        return int(self.fc.out_features)


def embed(windows, encoder, batch=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            x = torch.as_tensor(np.asarray(windows[i:i + batch]), dtype=torch.float32, device=device)
            out.append(encoder(x).cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, encoder.dim), np.float32)
