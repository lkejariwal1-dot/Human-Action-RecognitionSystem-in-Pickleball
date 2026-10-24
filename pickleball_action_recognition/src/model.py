import torch
import torch.nn as nn
from src.features import IN_CH


class PoseCNN(nn.Module):
    """Spatial module: 1-D convolutions over the (chain-ordered) joints of ONE frame."""

    def __init__(self, in_ch=IN_CH, emb=128, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),                                    # 17 -> 8 joints
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(128 * 8, emb), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class CNNBiLSTM(nn.Module):
    """Time-distributed PoseCNN -> 2-layer BiLSTM -> mean+max pooling -> classifier."""

    def __init__(self, num_classes, in_ch=IN_CH, emb=128, hidden=128, layers=2, p=0.3):
        super().__init__()
        self.cnn = PoseCNN(in_ch, emb, p)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, bidirectional=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden * 4, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, num_classes))

    def forward(self, x):                                       # x: (B, T, J, C)
        B, T, J, C = x.shape
        f = self.cnn(x.permute(0, 1, 3, 2).reshape(B * T, C, J)).reshape(B, T, -1)
        o, _ = self.lstm(f)
        return self.head(torch.cat([o.mean(1), o.max(1).values], 1))
