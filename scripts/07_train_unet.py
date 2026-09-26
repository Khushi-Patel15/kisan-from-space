"""Step 7: train the U-Net -> models/unet_best.pt  (run this on a GPU: Kaggle or Colab)

How it learns:
  - input: a 12-band satellite crop of 128x128 pixels (1.28 km) around some surveyed fields
  - target: the crop class of every surveyed TRAIN-field pixel (all other pixels are ignored)
  - loss: cross-entropy, with extra weight on rare crops so it doesn't just learn "wheat"
After every few epochs it predicts the VALIDATION fields (field = average of its pixels)
and keeps the model with the best macro F1. Test fields are never touched here.

Run from the project root:  python scripts/07_train_unet.py
Quick smoke test on a laptop CPU:  python scripts/07_train_unet.py --epochs 1 --max-chips 40 --no-pretrained
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import CLASS_IDS, IGNORE, MODELS  # noqa: E402
from kisan.dl import (ChipDataset, build_model, field_probabilities, field_splits,  # noqa: E402
                      load_packed, split_masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=128, help="training crop size in pixels")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--max-chips", type=int, default=None, help="use fewer chips (for a quick test)")
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="rare-crop boost: 0 = none, 0.5 = moderate (default), 1.0 = strong")
    ap.add_argument("--repeats", type=int, default=4, help="random crops per training chip per epoch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    MODELS.mkdir(exist_ok=True)

    images, labels, fields, chips, stats = load_packed()
    splits = field_splits()
    train_mask = split_masks(labels, fields, splits, "train")
    val_fields = {f for f, s in splits.items() if s == "val"}
    field_crop = pd.read_csv("data/fields.csv").set_index("field_id")["crop_id"].to_dict()

    train_idx = np.where((train_mask != IGNORE).any(axis=(1, 2)))[0]
    val_idx = np.where(np.isin(fields, list(val_fields)).any(axis=(1, 2)))[0]
    if args.max_chips:
        train_idx, val_idx = train_idx[:args.max_chips], val_idx[:args.max_chips // 2]
    print(f"Training chips: {len(train_idx)} · validation chips: {len(val_idx)}")

    # class weights: rarer crop -> bigger weight. weight = 1 / frequency ** power
    counts = np.bincount(train_mask[train_mask != IGNORE], minlength=len(CLASS_IDS)).astype(float)
    weights = 1 / np.maximum(counts, 1) ** args.weight_power
    weights = weights / weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                        ignore_index=IGNORE)

    ds = ChipDataset(images, train_mask, train_idx, stats, crop=args.crop, augment=True)
    # each epoch visits every training chip `repeats` times (a different random crop each time)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=len(ds) * args.repeats)
    loader = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=2, drop_last=True)

    model = build_model(args.encoder, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, history = -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
        row = {"epoch": epoch, "train_loss": total / len(loader), "seconds": time.time() - t0}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            probs = field_probabilities(model, images, fields, val_idx, stats, device, wanted=val_fields)
            ids = list(probs)
            y_true = [field_crop[f] for f in ids]
            y_pred = [CLASS_IDS[int(np.argmax(probs[f]))] for f in ids]
            row["val_accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
            row["val_macro_F1"] = f1_score(y_true, y_pred, average="macro", labels=CLASS_IDS, zero_division=0)
            if row["val_macro_F1"] > best:
                best = row["val_macro_F1"]
                torch.save({"state_dict": model.state_dict(), "encoder": args.encoder, "epoch": epoch,
                            "val_macro_F1": best}, MODELS / "unet_best.pt")
                row["saved"] = "*"
        history.append(row)
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        pd.DataFrame(history).to_csv(MODELS / "train_history.csv", index=False)

    print(f"\nBest validation macro F1: {best:.3f}  (model saved to {MODELS / 'unet_best.pt'})")


if __name__ == "__main__":
    main()