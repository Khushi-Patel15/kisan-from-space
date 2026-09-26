"""Step 5: baseline models -> outputs/baseline/

Two baselines, trained on the per-field features from step 2:
  1. "Always Wheat"  - predicts the most common crop for every field. Shows why plain
                       accuracy is misleading when classes are unbalanced.
  2. Random Forest   - a classic, strong ML model on field-average band values.
The deep-learning model in Week 2 has to beat the Random Forest.

Reported metrics:
  accuracy  - % of fields correct (flattered by the big classes)
  macro F1  - average F1 over crops, every crop counts equally (our MAIN metric)
Run from the project root:  python scripts/05_baseline.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import DATA, OUT  # noqa: E402

RES = OUT / "baseline"
SIZE_BINS = [0, 0.1, 0.25, 0.5, 1.0, np.inf]
SIZE_LABELS = ["<0.1 ha", "0.1-0.25", "0.25-0.5", "0.5-1", ">1 ha"]


def scores(y, p):
    return {"accuracy": accuracy_score(y, p), "macro_F1": f1_score(y, p, average="macro", zero_division=0),
            "weighted_F1": f1_score(y, p, average="weighted", zero_division=0)}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "fields.csv").merge(pd.read_csv(DATA / "splits.csv"), on="field_id")
    features = [c for c in df.columns if c.startswith(("mean_", "std_"))] + ["area_ha"]
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr)} · val {len(va)} · test {len(te)} fields · {len(features)} features")

    models = {
        "Always most common crop": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                min_samples_leaf=2, n_jobs=-1, random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(tr[features], tr["crop"])
        for split, d in (("val", va), ("test", te)):
            rows.append({"model": name, "split": split, **scores(d["crop"], model.predict(d[features]))})
    res = pd.DataFrame(rows)
    table = res.pivot(index="model", columns="split", values=["accuracy", "macro_F1"]).round(3)
    table.columns = [f"{metric} ({split})" for metric, split in table.columns]
    rf = models["Random Forest"]
    pred = rf.predict(te[features])

    # --- per-crop results (test) ---
    report = classification_report(te["crop"], pred, zero_division=0, output_dict=True)
    per_crop = pd.DataFrame(report).T.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    per_crop = per_crop.rename(columns={"support": "test_fields"}).round(3).sort_values("test_fields", ascending=False)

    # --- accuracy vs field size (test): the key analysis for the research paper ---
    te = te.assign(correct=(te["crop"] == pred), size_bin=pd.cut(te["area_ha"], SIZE_BINS, labels=SIZE_LABELS))
    by_size = te.groupby("size_bin", observed=False)["correct"].agg(fields="size", accuracy="mean")

    md = ["## Baseline results (field-level)", "", table.to_markdown(), "",
          "## Random Forest, per crop (test)", "", per_crop.to_markdown(), "",
          "## Random Forest accuracy by field size (test)", "", by_size.round(3).to_markdown()]
    (RES / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # --- figures ---
    labels = per_crop.index.tolist()
    cm = confusion_matrix(te["crop"], pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xlabel("Predicted crop"); ax.set_ylabel("True crop")
    ax.set_title("Random Forest confusion matrix (test, each row sums to 1)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(RES / "confusion_matrix.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_size.index.astype(str), by_size["accuracy"], color="#2D6A4F")
    for i, (acc, n) in enumerate(zip(by_size["accuracy"], by_size["fields"])):
        ax.text(i, acc + 0.01, f"{acc:.2f}\n(n={int(n)})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_xlabel("Field size")
    ax.set_title("Does the model struggle with small fields?")
    fig.tight_layout(); fig.savefig(RES / "accuracy_by_field_size.png", dpi=120); plt.close(fig)

    imp = pd.Series(rf.feature_importances_, index=features).sort_values()[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#B57A12"); ax.set_title("Top 15 features the Random Forest relies on")
    fig.tight_layout(); fig.savefig(RES / "feature_importance.png", dpi=120); plt.close(fig)
    print(f"\nSaved to {RES.resolve()}")


if __name__ == "__main__":
    main()