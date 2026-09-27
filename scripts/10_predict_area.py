"""Step 10: map the crops of a prepared area -> data/areas/<name>/

Runs the U-Net over the whole area in overlapping 256x256 tiles (overlaps are averaged so
there are no seams), keeps only cropland pixels, and writes:
  crop_map.tif     class index 0..12 per pixel, 255 = not cropland
  confidence.tif   model confidence 0-100
  preview.png      satellite image | crop map, side by side
  crop_areas.csv   hectares of each crop in the area
Run:  python scripts/10_predict_area.py --name nanpara
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import CLASS_NAMES, CROP_COLOURS, DATA, MODELS, PACKED, PIXEL_AREA_HA  # noqa: E402
from kisan.dl import build_model, predict_probs  # noqa: E402

TILE, STRIDE, NOT_CROP = 256, 192, 255


def tiled_probs(model, image, stats, device):
    """Class probabilities (13, H, W) for an image of any size, from overlapping tiles."""
    _, h, w = image.shape
    H, W = max(h, TILE), max(w, TILE)
    ys = list(range(0, H - TILE + 1, STRIDE)); xs = list(range(0, W - TILE + 1, STRIDE))
    if ys[-1] != H - TILE: ys.append(H - TILE)
    if xs[-1] != W - TILE: xs.append(W - TILE)
    padded = np.zeros((12, H, W), dtype=np.uint8); padded[:, :h, :w] = image
    tiles = np.stack([padded[:, y:y + TILE, x:x + TILE] for y in ys for x in xs])
    probs = np.zeros((13, H, W), dtype=np.float32); weight = np.zeros((H, W), dtype=np.float32)
    pos = [(y, x) for y in ys for x in xs]
    for i, p in predict_probs(model, tiles, np.arange(len(tiles)), stats, device):
        y, x = pos[i]
        probs[:, y:y + TILE, x:x + TILE] += p; weight[y:y + TILE, x:x + TILE] += 1
    return (probs / weight)[:, :h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    folder = DATA / "areas" / args.name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(MODELS / "unet_best.pt", map_location=device)
    model = build_model(ckpt["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    stats = np.load(PACKED / "band_stats.npy")

    with rasterio.open(folder / "image.tif") as src:
        image, profile = src.read(), src.profile
    with rasterio.open(folder / "cropland.tif") as src:
        cropland = src.read(1).astype(bool)

    probs = tiled_probs(model, image, stats, device)
    crop_map = probs.argmax(0).astype(np.uint8)
    conf = (probs.max(0) * 100).round().astype(np.uint8)
    crop_map[~cropland] = NOT_CROP

    profile.update(count=1, dtype="uint8", nodata=NOT_CROP)
    with rasterio.open(folder / "crop_map.tif", "w", **profile) as dst:
        dst.write(crop_map, 1)
    profile.update(nodata=None)
    with rasterio.open(folder / "confidence.tif", "w", **profile) as dst:
        dst.write(conf, 1)

    # area per crop
    counts = np.bincount(crop_map[cropland], minlength=13)
    table = pd.DataFrame({"crop": CLASS_NAMES, "hectares": (counts * PIXEL_AREA_HA).round(1),
                          "share_of_cropland": (counts / max(counts.sum(), 1)).round(3)})
    table = table[table["hectares"] > 0].sort_values("hectares", ascending=False)
    table.to_csv(folder / "crop_areas.csv", index=False)

    # preview: satellite | crop map
    info = json.loads((folder / "info.json").read_text())
    rgb = image[[3, 2, 1]].transpose(1, 2, 0).astype(float)
    for k in range(3):
        lo, hi = np.percentile(rgb[..., k], [1, 99.5]); rgb[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo + 1e-6), 0, 1)
    colours = np.array([matplotlib.colors.to_rgb(CROP_COLOURS[n]) for n in CLASS_NAMES])
    overlay = rgb * 0.35
    overlay[cropland] = colours[crop_map[cropland]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    axes[0].imshow(rgb); axes[0].set_title(f"Sentinel-2, {info['date']}")
    axes[1].imshow(overlay); axes[1].set_title("AI crop map (non-cropland dimmed)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    present = table["crop"].tolist()
    handles = [plt.Rectangle((0, 0), 1, 1, color=CROP_COLOURS[c]) for c in present]
    fig.legend(handles, [f"{c} ({h:g} ha)" for c, h in zip(table["crop"], table["hectares"])],
               loc="lower center", ncol=min(len(present), 7), frameon=False)
    fig.suptitle(f"{info['name']}: {info['size_km']:g} km x {info['size_km']:g} km around "
                 f"{info['lat']:.3f}N {info['lon']:.3f}E", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97)); fig.savefig(folder / "preview.png", dpi=110); plt.close(fig)

    print(table.to_string(index=False))
    print(f"\nMean confidence on cropland: {conf[cropland].mean():.0f}%   Saved to {folder}")


if __name__ == "__main__":
    main()