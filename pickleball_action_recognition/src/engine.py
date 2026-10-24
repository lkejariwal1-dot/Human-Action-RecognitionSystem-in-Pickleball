import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    ys, ps, probs, loss_sum, n = [], [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        n += len(y)
        pr = torch.softmax(logits, 1)
        probs.append(pr.cpu().numpy())
        ps += pr.argmax(1).cpu().tolist()
        ys += y.cpu().tolist()
    return dict(loss=loss_sum / n, acc=accuracy_score(ys, ps),
                f1=f1_score(ys, ps, average="macro", zero_division=0),
                y=np.array(ys), p=np.array(ps), probs=np.concatenate(probs))
