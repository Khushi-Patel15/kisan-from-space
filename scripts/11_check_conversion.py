"""Step 11: prove the Sentinel-2 conversion works -> outputs/week3/conversion_check.md

For some TEST-split chips we already know AgriFieldNet's real 8-bit images. For each one we:
  1. calibrate using OTHER nearby chips only (the test chip itself is excluded),
  2. download the fresh Sentinel-2 image of that chip and convert it,
  3. compare: band values (error in 8-bit units) and the U-Net's crop predictions
     on the real image vs the converted image.
If predictions agree closely, the model will behave the same on new areas we convert this way.
Run from the project root:  python scripts/11_check_conversion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import MODELS, OUT  # noqa: E402
from kisan.dl import build_model, load_packed, predict_probs  # noqa: E402
from kisan.s2 import calibrate, chip_index, read_bands, to_uint8  # noqa: E402

N_CHIPS = 8


def main():
    out = OUT / "week3"; out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODELS / "unet_best.pt", map_location=device)
    model = build_model(ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Using U-Net from epoch {ckpt['epoch']} (val macro F1 {ckpt['val_macro_F1']:.3f})")
    images, labels, fields, chips, stats = load_packed()

    # pick test chips with the most labelled fields, spread over states
    f = pd.read_csv("data/fields.csv").merge(pd.read_csv("data/splits.csv"), on="field_id")
    f = f[f["split"] == "test"].assign(chip=lambda d: d["chips"].str.split(";")).explode("chip")
    per_chip = f.groupby(["state", "chip"]).size().reset_index(name="n").sort_values("n", ascending=False)
    picks = per_chip.groupby("state").head(2).head(N_CHIPS)
    idx = chip_index().set_index("chip")

    rows = []
    for _, p in picks.iterrows():
        r = idx.loc[p["chip"]]
        try:
            item, coef, info = calibrate(r["lon"], r["lat"], exclude=[p["chip"]])
        except ValueError as e:
            print(f"{p['chip']}: skipped ({e})"); continue
        refl = read_bands(item, r["crs"], (r["left"], r["bottom"], r["right"], r["top"]))
        mine = to_uint8(refl, coef)
        i = chips.index(p["chip"])
        real = np.asarray(images[i])
        mae = np.abs(mine.astype(float) - real).mean(axis=(1, 2))

        # U-Net on the real image vs on our converted image
        stack = np.stack([real, mine])
        preds = [pr.argmax(0) for _, pr in predict_probs(model, stack, np.arange(2), stats, device)]
        fid = fields[i]
        on_fields = fid > 0
        field_agree = [np.bincount(preds[0][fid == k]).argmax() == np.bincount(preds[1][fid == k]).argmax()
                       for k in np.unique(fid[on_fields])]
        row = {"chip": p["chip"], "state": p["state"], "scene date": info["date"],
               "calibrated from": len(info["calibration_chips"]),
               "mean error (0-255)": round(float(mae.mean()), 2),
               "pixel agreement (fields)": round(float((preds[0][on_fields] == preds[1][on_fields]).mean()), 3),
               "field agreement": round(float(np.mean(field_agree)), 3), "fields": len(field_agree),
               # sanity check: a model that predicts one crop everywhere would "agree" trivially
               "crops predicted": len(np.unique(preds[0][on_fields]))}
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    md = ["## Conversion check: U-Net on real AgriFieldNet image vs our converted Sentinel-2 image", "",
          df.to_markdown(index=False), "",
          f"Average field agreement: {df['field agreement'].mean():.3f}"]
    (out / "conversion_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()