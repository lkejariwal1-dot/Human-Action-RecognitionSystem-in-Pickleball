import subprocess
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from src.model import CNNBiLSTM
from src.features import make_features

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
ANGLES = {"l_elbow": (5, 7, 9), "r_elbow": (6, 8, 10), "l_shoulder": (7, 5, 11), "r_shoulder": (8, 6, 12),
          "l_hip": (5, 11, 13), "r_hip": (6, 12, 14), "l_knee": (11, 13, 15), "r_knee": (12, 14, 16)}


def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def load_trained(path, device):
    ck = torch.load(path, map_location=device)
    m = CNNBiLSTM(len(ck["classes"]), in_ch=ck["in_ch"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["classes"]


@torch.no_grad()
def predict_clip(video_path, extractor, model, classes, device):
    out = extractor.extract_clip(video_path)
    if out is None:
        raise ValueError("No player detected in this video.")
    seq, _ = out
    x = torch.from_numpy(make_features(seq)).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0].cpu().numpy()
    return classes[int(probs.argmax())], float(probs.max()), dict(zip(classes, probs.round(4).tolist()))


def analyze_video(video_path, extractor, model, classes, device, out_dir, write_video=True):
    """Prediction + skeleton video + per-frame joint angles (CSV)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    action, conf, probs = predict_clip(video_path, extractor, model, classes, device)
    stem = Path(video_path).stem
    raw_path, final_path = out_dir / f"{stem}_raw.mp4", out_dir / f"{stem}_annotated.mp4"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)) if write_video else None
    rows, prev, fi = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = extractor.detect(frame, prev)
        row = {"frame": fi, "time_s": round(fi / fps, 3)}
        if p is not None:
            kxy, kc, box = p
            prev = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            ok_kp = kc > 0.3
            for name, (a, b, c) in ANGLES.items():
                row[name] = joint_angle(kxy[a], kxy[b], kxy[c]) if (ok_kp[a] and ok_kp[b] and ok_kp[c]) else np.nan
            for j, nm in [(5, "l_shoulder"), (6, "r_shoulder"), (9, "l_wrist"), (10, "r_wrist")]:
                if ok_kp[j]:
                    row[f"{nm}_x"], row[f"{nm}_y"] = float(kxy[j][0]), float(kxy[j][1])
            if writer is not None:
                for a, b in SKELETON:
                    if ok_kp[a] and ok_kp[b]:
                        cv2.line(frame, tuple(map(int, kxy[a])), tuple(map(int, kxy[b])), (0, 255, 0), 2)
                for j in range(17):
                    if ok_kp[j]:
                        cv2.circle(frame, tuple(map(int, kxy[j])), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (255, 160, 0), 2)
        if writer is not None:
            cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{action.upper()}  {conf * 100:.1f}%", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            writer.write(frame)
        rows.append(row)
        fi += 1
    cap.release()

    video_out = None
    if writer is not None:
        writer.release()
        try:                                     # re-encode so it plays in browsers / phones
            res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                                  "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(final_path)])
            ok_enc = res.returncode == 0
        except FileNotFoundError:                # ffmpeg not installed
            ok_enc = False
        if ok_enc:
            raw_path.unlink(missing_ok=True)
            video_out = final_path
        else:
            video_out = raw_path                 # mp4v file (plays in VLC)

    angles_df = pd.DataFrame(rows)
    csv_path = out_dir / f"{stem}_joint_angles.csv"
    angles_df.to_csv(csv_path, index=False)
    return dict(action=action, confidence=conf, probabilities=probs, video=video_out, angles_csv=csv_path, angles=angles_df)
