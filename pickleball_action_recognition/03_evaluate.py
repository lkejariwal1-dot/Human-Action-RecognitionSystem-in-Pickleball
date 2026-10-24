"""STEP 3: evaluate the best model on the held-out test set (accuracy, P/R/F1, confusion matrix)."""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader
import config
from src.data import PoseClipDataset
from src.engine import evaluate
from src.inference import load_trained
from src.utils import get_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project_dir", default=config.PROJECT_DIR)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    paths = config.make_paths(args.project_dir)
    device = get_device()
    model, classes = load_trained(paths.best, device)
    test_df = pd.read_csv(paths.root / "split_test.csv")
    loader = DataLoader(PoseClipDataset(test_df, paths.cache, False), batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=args.workers)

    t = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    labels = list(range(len(classes)))
    print(f"\nTest accuracy: {t['acc']:.4f} | macro-F1: {t['f1']:.4f}\n")
    print(classification_report(t["y"], t["p"], labels=labels, target_names=classes, digits=4, zero_division=0))

    cm = confusion_matrix(t["y"], t["p"], labels=labels)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax[0], cmap="Blues", xticks_rotation=45, colorbar=False)
    ax[0].set_title("Confusion matrix (counts)")
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ConfusionMatrixDisplay(cmn, display_labels=classes).plot(ax=ax[1], cmap="Blues", xticks_rotation=45,
                                                             colorbar=False, values_format=".2f")
    ax[1].set_title("Confusion matrix (row-normalised)")
    plt.tight_layout(); plt.savefig(paths.results / "confusion_matrix.png", dpi=150); plt.close()

    pr, rc, f1, sup = precision_recall_fscore_support(t["y"], t["p"], labels=labels, zero_division=0)
    metrics = dict(test_accuracy=t["acc"], macro_f1=t["f1"],
                   per_class={c: dict(precision=float(a), recall=float(b), f1=float(f), support=int(s))
                              for c, a, b, f, s in zip(classes, pr, rc, f1, sup)},
                   confusion_matrix=cm.tolist())
    json.dump(metrics, open(paths.results / "test_metrics.json", "w"), indent=2)

    wrong = test_df.assign(pred=[classes[i] for i in t["p"]], confidence=t["probs"].max(1))
    wrong = wrong[wrong.label != wrong.pred][["path", "label", "pred", "confidence"]]
    wrong.to_csv(paths.results / "misclassified_test_clips.csv", index=False)
    print(f"Misclassified test clips: {len(wrong)} (listed in misclassified_test_clips.csv)")
    print(f"Saved metrics + plots to {paths.results}")


if __name__ == "__main__":
    main()
