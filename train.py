"""Step 2: Train the CNN (MobileNetV2 transfer learning) on dataset/<class>/ images.

Usage:
    python train.py                 # default: 8 epochs head + 7 epochs fine-tune
    python train.py --epochs1 5 --epochs2 5     # faster

Outputs:
    models/compost_cnn.pt                 trained weights + class list
    results/training_curves.png           loss/accuracy per epoch
    results/confusion_matrix.png          validation confusion matrix
    results/classification_report.txt     precision / recall / F1 per stage
    results/metrics.json                  summary numbers for the slides
"""

import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from config import (BATCH_SIZE, CLASS_LABELS, CLASSES, DATA_DIR, IMG_SIZE,
                    MODEL_DIR, RESULTS_DIR, SEED)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.25, 0.25, 0.15, 0.03),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Subset(Dataset):
    def __init__(self, base, idx, tf):
        self.base, self.idx, self.tf = base, idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        path, label = self.base.samples[self.idx[i]]
        img = self.base.loader(path).convert("RGB")
        return self.tf(img), label


def build_model(n_classes, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, n_classes))
    return m


def run_epoch(model, loader, crit, opt, device, train):
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            p = out.argmax(1)
            tot_loss += loss.item() * len(y)
            correct += (p == y).sum().item()
            n += len(y)
            preds_all += p.cpu().tolist()
            labels_all += y.cpu().tolist()
    return tot_loss / n, correct / n, preds_all, labels_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs1", type=int, default=8, help="epochs training only the head")
    ap.add_argument("--epochs2", type=int, default=7, help="epochs fine-tuning last blocks")
    ap.add_argument("--no-pretrained", action="store_true", help="(testing only)")
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    base = datasets.ImageFolder(DATA_DIR)
    assert base.classes == CLASSES, f"dataset/ folders must be exactly {CLASSES}, found {base.classes}"
    labels = [l for _, l in base.samples]
    counts = np.bincount(labels, minlength=len(CLASSES))
    for c, k in zip(CLASSES, counts):
        print(f"  {c}: {k} images")

    tr_idx, va_idx = train_test_split(range(len(labels)), test_size=0.2,
                                      stratify=labels, random_state=SEED)
    tr_dl = DataLoader(Subset(base, tr_idx, train_tf), BATCH_SIZE, shuffle=True, num_workers=0)
    va_dl = DataLoader(Subset(base, va_idx, val_tf), BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(tr_idx)}  Val: {len(va_idx)}")

    model = build_model(len(CLASSES), pretrained=not args.no_pretrained).to(device)
    # class weights handle imbalanced folders
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_eval = -1, None, None

    def train_phase(epochs, opt, name):
        nonlocal best_acc, best_state, best_eval
        for ep in range(1, epochs + 1):
            t = time.time()
            tl, ta, _, _ = run_epoch(model, tr_dl, crit, opt, device, True)
            vl, va, vp, vy = run_epoch(model, va_dl, crit, None, device, False)
            for k, v in zip(hist, (tl, vl, ta, va)):
                hist[k].append(v)
            print(f"[{name}] epoch {ep}/{epochs}  train_acc={ta:.3f}  val_acc={va:.3f}  "
                  f"val_loss={vl:.3f}  ({time.time()-t:.0f}s)")
            if va >= best_acc:
                best_acc = va
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_eval = (vp, vy)

    # Phase 1: freeze backbone, train classifier head
    for p in model.features.parameters():
        p.requires_grad = False
    train_phase(args.epochs1, torch.optim.Adam(model.classifier.parameters(), lr=1e-3), "head")

    # Phase 2: unfreeze last 4 feature blocks, fine-tune with low LR
    for p in model.features[-4:].parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    train_phase(args.epochs2, torch.optim.Adam(params, lr=1e-4), "finetune")

    MODEL_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "classes": CLASSES}, MODEL_DIR / "compost_cnn.pt")

    # ---- results for the presentation ----
    vp, vy = best_eval
    names = [CLASS_LABELS[c].split(" - ")[1] for c in CLASSES]
    report = classification_report(vy, vp, labels=range(len(CLASSES)), target_names=names, digits=3, zero_division=0)
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    print("\n" + report)

    cm = confusion_matrix(vy, vp, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (val acc {best_acc:.1%})")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)

    ep = range(1, len(hist["val_acc"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["train_loss"], label="train"); a1.plot(ep, hist["val_loss"], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, hist["train_acc"], label="train"); a2.plot(ep, hist["val_acc"], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    a2.axvline(args.epochs1 + 0.5, ls="--", c="gray"); a2.text(args.epochs1 + 0.6, 0.05, "fine-tune starts", color="gray")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160); plt.close(fig)

    json.dump({"best_val_accuracy": round(best_acc, 4), "train_images": len(tr_idx),
               "val_images": len(va_idx), "per_class_counts": dict(zip(CLASSES, counts.tolist())),
               "model": "MobileNetV2 (ImageNet pretrained)", "device": device},
              open(RESULTS_DIR / "metrics.json", "w"), indent=2)
    print(f"Best validation accuracy: {best_acc:.1%}")
    print("Saved model to models/compost_cnn.pt and plots to results/")


if __name__ == "__main__":
    main()
