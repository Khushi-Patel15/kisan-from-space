"""Step 8: evaluate the U-Net against the Random Forest -> outputs/week2/

  1. Runs the trained U-Net over ALL chips once and saves the crop map + confidence
     for every pixel (data/predictions/) - the Week 4 website displays these.
  2. Turns pixel predictions into field predictions (average over each field's pixels).
  3. Compares on the SAME validation and test fields:
       Random Forest (step 5) · U-Net · Ensemble (average of both models' probabilities)
  4. Per-crop scores and the field-size analysis for the report.
Run from the project root (GPU recommended, CPU works but takes ~15 min):
  python scripts/08_evaluate.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import CLASS_IDS, CROPS, DATA, MODELS, OUT  # noqa: E402
from kisan.dl import build_model, load_packed, predict_probs  # noqa: E402

RES = OUT / "week2"
PRED = DATA / "predictions"
SIZE_BINS = [0, 0.1, 0.25, 0.5, 1.0, np.inf]
SIZE_LABELS = ["<0.1 ha", "0.1-0.25", "0.25-0.5", "0.5-1", ">1 ha"]


def unet_field_probs(device):
    """Predict every chip; save pixel maps; return {field_id: mean class probabilities}."""
    ckpt = torch.load(MODELS / "unet_best.pt", map_location=device)
    model = build_model(ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded U-Net from epoch {ckpt['epoch']} (val macro F1 {ckpt['val_macro_F1']:.3f})")

    images, labels, fields, chips, stats = load_packed()
    PRED.mkdir(parents=True, exist_ok=True)
    pred_class = np.lib.format.open_memmap(PRED / "pred_class.npy", mode="w+", dtype=np.uint8, shape=fields.shape)
    pred_conf = np.lib.format.open_memmap(PRED / "pred_conf.npy", mode="w+", dtype=np.uint8, shape=fields.shape)
    sums, counts = {}, {}
    for i, probs in tqdm(predict_probs(model, images, np.arange(len(chips)), stats, device),
                         total=len(chips), unit="chip"):
        pred_class[i] = probs.argmax(0)
        pred_conf[i] = (probs.max(0) * 100).round()
        fid = fields[i]
        for f in np.unique(fid[fid > 0]):
            m = fid == f
            sums[f] = sums.get(f, 0) + probs[:, m].sum(1)
            counts[f] = counts.get(f, 0) + m.sum()
    pred_class.flush(); pred_conf.flush()
    (PRED / "chips.txt").write_text("\n".join(chips))
    return {int(f): sums[f] / counts[f] for f in sums}


def rf_probs(df):
    """Retrain the step-5 Random Forest; return class probabilities for every field."""
    features = [c for c in df.columns if c.startswith(("mean_", "std_"))] + ["area_ha"]
    tr = df[df["split"] == "train"]
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                min_samples_leaf=2, n_jobs=-1, random_state=42)
    rf.fit(tr[features], tr["crop_id"])
    p = rf.predict_proba(df[features])
    out = np.zeros((len(df), len(CLASS_IDS)))
    for j, c in enumerate(rf.classes_):          # align columns to CLASS_IDS order
        out[:, CLASS_IDS.index(c)] = p[:, j]
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(DATA / "fields.csv").merge(pd.read_csv(DATA / "splits.csv"), on="field_id")

    unet = unet_field_probs(device)
    P = {"Random Forest": rf_probs(df),
         "U-Net": np.stack([unet[f] for f in df["field_id"]])}
    P["Ensemble (RF + U-Net)"] = (P["Random Forest"] + P["U-Net"]) / 2
    for name, p in P.items():
        df[name] = [CLASS_IDS[k] for k in p.argmax(1)]
    df["U-Net confidence"] = P["U-Net"].max(1).round(3)
    df[["field_id", "split", "crop_id", "crop", "area_ha", "state", *P, "U-Net confidence"]].to_csv(
        RES / "field_predictions.csv", index=False)

    # --- 1. headline comparison ---
    rows = []
    for name in P:
        for split in ("val", "test"):
            d = df[df["split"] == split]
            rows.append({"model": name, "split": split, "accuracy": accuracy_score(d["crop_id"], d[name]),
                         "macro_F1": f1_score(d["crop_id"], d[name], average="macro", labels=CLASS_IDS, zero_division=0)})
    table = pd.DataFrame(rows).pivot(index="model", columns="split", values=["accuracy", "macro_F1"]).round(3)
    table.columns = [f"{m} ({s})" for m, s in table.columns]
    table = table.reindex(list(P))

    te = df[df["split"] == "test"].copy()
    # --- 2. per crop (test) ---
    per_crop = pd.DataFrame({name: f1_score(te["crop_id"], te[name], average=None, labels=CLASS_IDS, zero_division=0)
                             for name in P}, index=CLASS_IDS)
    per_crop.insert(0, "test_fields", [int((te["crop_id"] == c).sum()) for c in CLASS_IDS])
    per_crop.index = [CROPS[c] for c in CLASS_IDS]
    per_crop = per_crop.sort_values("test_fields", ascending=False).round(3)

    # --- 3. field size (test). Balanced accuracy = average recall over the crops present
    #        in that size group, so a group full of easy fallow fields can't look artificially good.
    te["size_bin"] = pd.cut(te["area_ha"], SIZE_BINS, labels=SIZE_LABELS)
    size_rows = []
    for b in SIZE_LABELS:
        g = te[te["size_bin"] == b]
        r = {"size": b, "fields": len(g)}
        for name in P:
            r[f"{name} accuracy"] = accuracy_score(g["crop_id"], g[name])
            r[f"{name} balanced acc."] = balanced_accuracy_score(g["crop_id"], g[name])
        size_rows.append(r)
    by_size = pd.DataFrame(size_rows).set_index("size").round(3)

    md = ["## Week 2 results: field-level crop classification", "", table.to_markdown(), "",
          "## F1 per crop (test)", "", per_crop.to_markdown(), "",
          "## By field size (test)", "", by_size.to_markdown()]
    (RES / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # --- figures ---
    best = table["macro_F1 (val)"].idxmax()   # choose the final model on VALIDATION, never on test
    names = per_crop.index.tolist()
    ids = [CLASS_IDS[[CROPS[c] for c in CLASS_IDS].index(n)] for n in names]
    cm = confusion_matrix(te["crop_id"], te[best], labels=ids, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.nan_to_num(cm), cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xlabel("Predicted crop"); ax.set_ylabel("True crop"); ax.set_title(f"{best}: confusion matrix (test)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(RES / "confusion_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(P)
    colours = ["#8C8C8C", "#2D6A4F", "#B57A12"]
    for k, name in enumerate(P):
        ax.bar(np.arange(len(SIZE_LABELS)) + k * w - 0.4 + w / 2, by_size[f"{name} balanced acc."], w, label=name, color=colours[k])
    ax.set_xticks(range(len(SIZE_LABELS)), [f"{s}\n(n={n})" for s, n in zip(SIZE_LABELS, by_size["fields"])])
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("How small is too small? Balanced accuracy by field size (test)")
    fig.tight_layout(); fig.savefig(RES / "accuracy_by_field_size.png", dpi=120); plt.close(fig)

    hist = pd.read_csv(MODELS / "train_history.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], color="#8C8C8C", label="train loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
    ax2 = ax1.twinx(); v = hist.dropna(subset=["val_macro_F1"])
    ax2.plot(v["epoch"], v["val_macro_F1"], "o-", color="#2D6A4F", label="val macro F1"); ax2.set_ylabel("val macro F1")
    ax1.set_title("U-Net training curve"); fig.tight_layout(); fig.savefig(RES / "training_curve.png", dpi=120); plt.close(fig)
    print(f"\nBest model on validation: {best}. Saved to {RES.resolve()}")


if __name__ == "__main__":
    main()