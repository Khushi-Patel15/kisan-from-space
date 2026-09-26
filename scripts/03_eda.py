"""Step 3: explore the data (EDA) -> figures and tables in outputs/eda/

Answers the questions your report needs:
  - How many fields of each crop are there? (class imbalance)
  - How small are the fields?            (the core challenge of this project)
  - Where are they?                       (fields per state)
  - Do crops look different to the satellite? (spectral signatures, NDVI)
Run from the project root:  python scripts/03_eda.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import BANDS, CHIPS, CROP_COLOURS, CROPS, DATA, OUT  # noqa: E402

EDA = OUT / "eda"
# Sentinel-2 band centre wavelengths (nm), used as the x-axis of the spectral plot
WAVELENGTH = {"B01": 443, "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740, "B07": 783,
              "B08": 842, "B8A": 865, "B09": 945, "B11": 1610, "B12": 2190}


def crop_counts(df, lines):
    counts = df["crop"].value_counts()
    area = df.groupby("crop")["area_ha"].sum().reindex(counts.index)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=[CROP_COLOURS[c] for c in counts.index[::-1]],
            edgecolor="#333", linewidth=0.5)
    for i, (c, v) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
        ax.text(v, i, f"  {v} ({v / len(df):.1%})", va="center", fontsize=9)
    ax.set_xlabel("Number of fields"); ax.set_title("Fields per crop (training labels)")
    ax.set_xlim(0, counts.max() * 1.25)
    fig.tight_layout(); fig.savefig(EDA / "crop_counts.png", dpi=120); plt.close(fig)
    table = pd.DataFrame({"fields": counts, "share_of_fields": (counts / len(df)).round(3),
                          "total_area_ha": area.round(1), "median_field_ha": df.groupby("crop")["area_ha"].median().round(2)}).reindex(counts.index)
    table.to_csv(EDA / "crop_table.csv")
    lines += ["Fields per crop:", table.to_string(), ""]


def field_sizes(df, lines):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(np.log10(0.01), np.log10(df["area_ha"].max()), 40)
    ax.hist(df["area_ha"], bins=bins, color="#2D6A4F", edgecolor="white")
    ax.set_xscale("log")
    med = df["area_ha"].median()
    ax.axvline(med, color="#B8432F", ls="--"); ax.text(med, ax.get_ylim()[1] * 0.92, f"  median {med:.2f} ha", color="#B8432F")
    ax.set_xlabel("Field area (hectares, log scale)   ·   1 pixel = 0.01 ha"); ax.set_ylabel("Number of fields")
    ax.set_title("How small are the fields?")
    fig.tight_layout(); fig.savefig(EDA / "field_sizes.png", dpi=120); plt.close(fig)
    q = df["area_ha"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    lines += ["Field size (ha) percentiles:", q.round(3).to_string(),
              f"Fields smaller than 0.1 ha (10 pixels): {(df['area_ha'] < 0.1).mean():.1%}",
              f"Fields smaller than 0.25 ha: {(df['area_ha'] < 0.25).mean():.1%}", ""]


def states(df, lines):
    t = pd.crosstab(df["crop"], df["state"], margins=True, margins_name="Total")
    t.to_csv(EDA / "crop_by_state.csv")
    lines += ["Fields per crop per state:", t.to_string(), ""]


def spectral_signatures(df):
    top = df["crop"].value_counts().index[:7]
    order = sorted(BANDS, key=WAVELENGTH.get)
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in top:
        sub = df[df["crop"] == c]
        ax.plot([WAVELENGTH[b] for b in order], [sub[f"mean_{b}"].mean() for b in order],
                marker="o", label=f"{c} (n={len(sub)})", color=CROP_COLOURS[c], lw=2)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Mean band value (8-bit)")
    ax.set_title("Spectral signature: how each crop reflects light, averaged over fields")
    ax.axvspan(700, 900, color="#999", alpha=0.08); ax.text(705, ax.get_ylim()[1] * 0.97, "red edge / near-infrared", fontsize=8, va="top")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(EDA / "spectral_signatures.png", dpi=120); plt.close(fig)


def ndvi_by_crop(df):
    order = df.groupby("crop")["mean_NDVI"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot([df.loc[df["crop"] == c, "mean_NDVI"] for c in order],
                    patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), order)
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(CROP_COLOURS[c])
    ax.set_ylabel("Field mean NDVI (greenness)"); ax.set_title("Greenness by crop on the image date")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(EDA / "ndvi_by_crop.png", dpi=120); plt.close(fig)


def gallery(df):
    """Six chips with the most labelled fields: true-colour image + fields outlined by crop."""
    chip_counts = df["chips"].str.split(";").explode().value_counts()
    picks = chip_counts.index[:6]
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
    for i, chip in enumerate(picks):
        folder = CHIPS / chip
        rgb = np.stack([rasterio.open(folder / f"{b}.tif").read(1) for b in ("B04", "B03", "B02")], -1).astype(float)
        for k in range(3):  # stretch each band so the image isn't dark
            lo, hi = np.percentile(rgb[..., k], [1, 99.5])
            rgb[..., k] = np.clip((rgb[..., k] - lo) / (hi - lo + 1e-6), 0, 1)
        label = rasterio.open(folder / "label.tif").read(1)
        overlay = np.zeros((*label.shape, 4))
        for cid, name in CROPS.items():
            col = matplotlib.colors.to_rgba(CROP_COLOURS[name])
            overlay[label == cid] = col
        axes[0, i].imshow(rgb); axes[0, i].set_title(f"chip {chip}", fontsize=9)
        axes[1, i].imshow(rgb * 0.5); axes[1, i].imshow(overlay)
        for ax in axes[:, i]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Satellite (true colour)"); axes[1, 0].set_ylabel("Surveyed fields by crop")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CROP_COLOURS[c]) for c in df["crop"].value_counts().index]
    fig.legend(handles, df["crop"].value_counts().index, loc="lower center", ncol=13, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(EDA / "gallery.png", dpi=110); plt.close(fig)


def main():
    EDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "fields.csv")
    lines = [f"Total labelled fields: {len(df)}", f"Total labelled area: {df['area_ha'].sum():.0f} ha", ""]
    crop_counts(df, lines)
    field_sizes(df, lines)
    states(df, lines)
    spectral_signatures(df)
    ndvi_by_crop(df)
    gallery(df)
    text = "\n".join(lines)
    (EDA / "summary.txt").write_text(text)
    print(text)
    print(f"Figures saved in {EDA.resolve()}")


if __name__ == "__main__":
    main()