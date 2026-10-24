"""STEP 2: stratified 70/15/15 split, train CNN-BiLSTM, keep the best model (val macro-F1)."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import config
from src.data import PoseClipDataset
from src.engine import evaluate
from src.model import CNNBiLSTM
from src.features import IN_CH
from src.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project_dir", default=config.PROJECT_DIR)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--patience", type=int, default=config.PATIENCE)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows; use 2 on Colab)")
    args = ap.parse_args()

    paths = config.make_paths(args.project_dir)
    set_seed(config.SEED)
    device = get_device()
    classes = config.CLASSES

    meta = pd.read_csv(paths.root / "meta.csv")
    min_count = meta.label.value_counts().min()
    if meta.label.nunique() < len(classes):
        print("WARNING: classes with no usable clips:", set(classes) - set(meta.label))
    if min_count < 7: # make it 7 after running
        raise ValueError(f"Smallest class has only {min_count} usable clips; need >= 7 (ideally 30+).")

    train_df, tmp = train_test_split(meta, test_size=0.30, stratify=meta.label_id, random_state=config.SEED)
    val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp.label_id, random_state=config.SEED)
    train_df, val_df, test_df = [d.reset_index(drop=True) for d in (train_df, val_df, test_df)]
    for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.to_csv(paths.root / f"split_{n}.csv", index=False)
    print(pd.DataFrame({n: d.label.value_counts() for n, d in
                        [("train", train_df), ("val", val_df), ("test", test_df)]}).reindex(classes).fillna(0).astype(int))
    # NOTE: if several clips come from the SAME source match video, split by source video instead,
    # otherwise near-duplicate clips leak between train and test and inflate accuracy.

    mk = lambda d, aug, shuf: DataLoader(PoseClipDataset(d, paths.cache, aug), batch_size=args.batch_size,
                                         shuffle=shuf, num_workers=args.workers,
                                         drop_last=shuf and len(d) > args.batch_size)
    train_loader, val_loader = mk(train_df, True, True), mk(val_df, False, False)

    model = CNNBiLSTM(len(classes)).to(device)
    print("Trainable parameters: %.2fM" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))

    counts = np.bincount(train_df.label_id, minlength=len(classes))
    cls_w = counts.sum() / (np.maximum(counts, 1) * len(classes))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32, device=device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_f1, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * len(y)
            tc += (logits.argmax(1) == y).sum().item()
            tn += len(y)
        scheduler.step()
        v = evaluate(model, val_loader, criterion, device)
        history.append(dict(epoch=epoch, train_loss=tl / tn, train_acc=tc / tn,
                            val_loss=v["loss"], val_acc=v["acc"], val_f1=v["f1"]))
        flag = ""
        if v["f1"] > best_f1:
            best_f1, bad, flag = v["f1"], 0, "  <- best"
            torch.save(dict(state_dict=model.state_dict(), classes=classes, seq_len=config.SEQ_LEN,
                            in_ch=IN_CH, epoch=epoch, val_f1=best_f1), paths.best)
        else:
            bad += 1
        print(f"Epoch {epoch:3d} | train loss {tl/tn:.3f} acc {tc/tn:.3f} | "
              f"val loss {v['loss']:.3f} acc {v['acc']:.3f} F1 {v['f1']:.3f}{flag}")
        # if bad >= args.patience:
        #     print(f"Early stopping (no val F1 improvement for {args.patience} epochs).")
        #     break

    hist = pd.DataFrame(history)
    hist.to_csv(paths.results / "training_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist.epoch, hist.train_loss, label="train"); ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.train_acc, label="train acc"); ax[1].plot(hist.epoch, hist.val_acc, label="val acc")
    ax[1].plot(hist.epoch, hist.val_f1, label="val macro-F1"); ax[1].set_title("Accuracy / F1"); ax[1].legend()
    plt.tight_layout(); plt.savefig(paths.results / "training_curves.png", dpi=150); plt.close()
    print(f"\nBest val macro-F1: {best_f1:.3f}. Model saved to {paths.best}. Next: python 03_evaluate.py")


if __name__ == "__main__":
    main()
