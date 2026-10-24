import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import VIDEO_EXT
from src.features import make_features


def scan_dataset(root, classes):
    """One sub-folder per action (case-insensitive). Returns DataFrame[path, rel, label, label_id]."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Check --dataset_root.")
    folders = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    rows = []
    for cid, c in enumerate(classes):
        if c.lower() not in folders:
            print(f'WARNING: no folder found for class "{c}" in {root}')
            continue
        vids = sorted(f for f in folders[c.lower()].rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXT)
        rows += [dict(path=str(v), rel=v.relative_to(root).as_posix(), label=c, label_id=cid) for v in vids]
    return pd.DataFrame(rows)


def cache_name(rel):
    """Cache file name derived from the path relative to the dataset root (portable across machines)."""
    return f"{Path(rel).stem}_{hashlib.md5(rel.encode()).hexdigest()[:8]}.npz"


class PoseClipDataset(Dataset):
    def __init__(self, frame, cache_dir, augment=False):
        self.files = [Path(cache_dir) / f for f in frame.cache_file]
        self.labels = frame.label_id.tolist()
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        seq = np.load(self.files[i])["seq"]
        return torch.from_numpy(make_features(seq, self.augment)), torch.tensor(self.labels[i])
