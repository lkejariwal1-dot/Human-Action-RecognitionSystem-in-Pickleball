"""STEP 1: scan the dataset, run YOLOv8-Pose on every clip, cache keypoints, write meta.csv."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from src.data import scan_dataset, cache_name
from src.pose import PoseExtractor
from src.utils import get_device


def main():
    ap = config.add_path_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--weights", default=config.POSE_WEIGHTS)
    ap.add_argument("--min_detect_rate", type=float, default=config.MIN_DETECT_RATE)
    args = ap.parse_args()
    paths = config.make_paths(args.project_dir)

    df = scan_dataset(args.dataset_root, config.CLASSES)
    assert len(df) > 0, "No videos found. Check --dataset_root and the folder names."
    print(f"Total clips: {len(df)}")
    print(df.label.value_counts().reindex(config.CLASSES).fillna(0).astype(int).to_string())

    extractor = PoseExtractor(args.weights, get_device())
    records, failed = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="Pose extraction"):
        fname = cache_name(row.rel)
        cp = paths.cache / fname
        try:
            if cp.exists():
                det = np.load(cp)["det"]
            else:
                out = extractor.extract_clip(row.path)
                if out is None:
                    failed.append((row.path, "unreadable or no person detected"))
                    continue
                seq, det = out
                np.savez_compressed(cp, seq=seq, det=det)
            records.append(dict(path=row.path, rel=row.rel, label=row.label, label_id=row.label_id,
                                cache_file=fname, detect_rate=float(det.mean())))
        except Exception as e:
            failed.append((row.path, repr(e)))

    meta_all = pd.DataFrame(records)
    meta = meta_all[meta_all.detect_rate >= args.min_detect_rate].reset_index(drop=True)
    meta_all.to_csv(paths.root / "meta_all.csv", index=False)
    meta.to_csv(paths.root / "meta.csv", index=False)
    print(f"\nUsable clips: {len(meta)} / {len(df)}  |  low detection rate: {len(meta_all) - len(meta)}  |  failed: {len(failed)}")
    for p, why in failed[:10]:
        print("  FAILED", Path(p).name, "->", why)
    print(meta.label.value_counts().reindex(config.CLASSES).fillna(0).astype(int).to_string())
    print(f"\nSaved {paths.root / 'meta.csv'}. Next: python 02_train.py")


if __name__ == "__main__":
    main()
