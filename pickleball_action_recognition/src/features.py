import random
import numpy as np
from config import KP_THR

# COCO keypoint ids ordered along the body: L-wrist -> head -> R-wrist -> R-leg -> L-leg
CHAIN = [9, 7, 5, 3, 1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 15, 13, 11]
# left <-> right swap (used for the horizontal-flip augmentation)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
IN_CH = 5   # x, y, confidence, velocity-x, velocity-y


def make_features(seq, augment=False):
    """seq: (T,17,3) in COCO order [x, y, conf]  ->  (T,17,5) in body-chain order."""
    seq = seq.copy().astype(np.float32)
    if augment:
        if random.random() < 0.5:                              # flip (handedness)
            seq = seq[:, FLIP_IDX]
            seq[..., 0] *= -1
        th = np.deg2rad(random.uniform(-10, 10))               # small rotation + scale
        s = random.uniform(0.9, 1.1)
        c, sn = np.cos(th), np.sin(th)
        x = seq[..., 0] * c - seq[..., 1] * sn
        y = seq[..., 0] * sn + seq[..., 1] * c
        seq[..., 0], seq[..., 1] = x * s, y * s
        vis = (seq[..., 2:3] > KP_THR).astype(np.float32)      # jitter noise on visible joints
        seq[..., :2] += np.random.normal(0, 0.01, seq[..., :2].shape).astype(np.float32) * vis
        seq[np.random.rand(*seq.shape[:2]) < 0.03] = 0.0       # random joint dropout (occlusion)
    seq = seq[:, CHAIN]
    vel = np.zeros_like(seq[..., :2])
    vel[1:] = seq[1:, :, :2] - seq[:-1, :, :2]
    return np.concatenate([seq, vel * 5.0], -1).astype(np.float32)
