import cv2
import numpy as np
from config import KP_THR, POSE_CONF, SEQ_LEN


def resize_max(frame, max_side=960):
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame


def sample_frames(path, seq_len=SEQ_LEN, max_side=960):
    """Uniformly sample seq_len frames (sequential read: robust against inaccurate seeking)."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < 1:                                   # unknown length -> read everything
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(f)
        cap.release()
        if not allf:
            return []
        idx = np.linspace(0, len(allf) - 1, seq_len).round().astype(int)
        return [resize_max(allf[i], max_side) for i in idx]
    idx = np.linspace(0, n - 1, seq_len).round().astype(int)
    want, got = set(idx.tolist()), {}
    for i in range(int(idx[-1]) + 1):
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                got[i] = resize_max(f, max_side)
    cap.release()
    if not got:
        return []
    avail = np.array(sorted(got))
    return [got[int(avail[np.abs(avail - i).argmin()])] for i in idx]


def pick_person(res, prev_center, W, H):
    """Main player = large + confident + close to the previous position. Returns (kxy, kconf, box) or None."""
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    bconf = res.boxes.conf.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy()                      # (N,17,2)
    kc = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
          else np.ones(kxy.shape[:2], dtype=np.float32))      # (N,17)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    score = area * bconf
    if prev_center is not None:
        ctr = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], 1)
        dist = np.linalg.norm(ctr - prev_center, axis=1) / np.hypot(W, H)
        score = score / (1 + 5 * dist)
    i = int(score.argmax())
    return kxy[i], kc[i], xyxy[i]


def normalize_pose(kxy, kc, box):
    """-> (17,3) = [x, y, conf]; hip-centred and divided by the player's box height."""
    vis = kc > KP_THR
    if vis[11] and vis[12]:
        centre = (kxy[11] + kxy[12]) / 2
    else:
        centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(float(box[3] - box[1]), 1.0)
    xy = (kxy - centre) / scale
    xy[~vis] = 0.0
    return np.concatenate([xy, kc[:, None]], 1).astype(np.float32)


class PoseExtractor:
    """Wraps YOLOv8-Pose. Loaded once and reused for every clip."""

    def __init__(self, weights, device, conf=POSE_CONF, seq_len=SEQ_LEN):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.dev = 0 if device.type == "cuda" else "cpu"
        self.conf, self.seq_len = conf, seq_len

    def detect(self, frame, prev_center=None):
        r = self.model.predict(frame, imgsz=640, conf=self.conf, device=self.dev, verbose=False)[0]
        H, W = frame.shape[:2]
        return pick_person(r, prev_center, W, H)

    def frames_to_pose(self, frames):
        results = self.model.predict(frames, imgsz=640, conf=self.conf, device=self.dev, verbose=False)
        T = len(frames)
        seq = np.zeros((T, 17, 3), np.float32)
        det = np.zeros(T, bool)
        prev = None
        for t, (f, r) in enumerate(zip(frames, results)):
            H, W = f.shape[:2]
            p = pick_person(r, prev, W, H)
            if p is None:
                continue
            kxy, kc, box = p
            seq[t] = normalize_pose(kxy, kc, box)
            det[t] = True
            prev = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if not det.any():
            return None
        idx = np.where(det)[0]
        for t in range(T):                       # fill gaps with the nearest detected frame
            if not det[t]:
                seq[t] = seq[idx[np.abs(idx - t).argmin()]]
        return seq, det

    def extract_clip(self, path):
        frames = sample_frames(path, self.seq_len)
        return self.frames_to_pose(frames) if frames else None
