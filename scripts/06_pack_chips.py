"""Step 6: pack all 1,165 chips into 4 big NumPy files -> data/packed/

Reading 16,000 small .tif files every training epoch is slow. Packing them once into
single arrays makes training fast (the arrays are memory-mapped, not loaded into RAM).

  images.npy   (N, 12, 256, 256) uint8   the 12 Sentinel-2 bands
  labels.npy   (N, 256, 256)     uint8   class index 0..12, 255 = unlabelled
  fields.npy   (N, 256, 256)     uint16  field ID, 0 = not a surveyed field
  chips.txt    chip ID for each of the N rows
  band_stats.npy (2, 12)         per-band mean and std, used to normalise the input
Run from the project root:  python scripts/06_pack_chips.py
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import BANDS, CHIPS, CLASS_IDS, IGNORE, PACKED  # noqa: E402


def main():
    PACKED.mkdir(parents=True, exist_ok=True)
    chips = sorted(p.name for p in CHIPS.iterdir() if p.is_dir())
    n = len(chips)
    images = np.lib.format.open_memmap(PACKED / "images.npy", mode="w+", dtype=np.uint8, shape=(n, 12, 256, 256))
    labels = np.full((n, 256, 256), IGNORE, dtype=np.uint8)
    fields = np.zeros((n, 256, 256), dtype=np.uint16)
    crop_to_index = {c: i for i, c in enumerate(CLASS_IDS)}

    for i, c in enumerate(tqdm(chips, unit="chip")):
        folder = CHIPS / c
        for b, band in enumerate(BANDS):
            with rasterio.open(folder / f"{band}.tif") as src:
                images[i, b] = src.read(1)
        with rasterio.open(folder / "label.tif") as src:
            lab = src.read(1)
        with rasterio.open(folder / "field_ids.tif") as src:
            fields[i] = src.read(1)
        for crop_id, idx in crop_to_index.items():
            labels[i][lab == crop_id] = idx

    images.flush()
    np.save(PACKED / "labels.npy", labels)
    np.save(PACKED / "fields.npy", fields)
    (PACKED / "chips.txt").write_text("\n".join(chips))

    # per-band mean/std (from image pixels only, no labels involved)
    sample = images[:, :, ::4, ::4].astype(np.float32)
    stats = np.stack([sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3))])
    np.save(PACKED / "band_stats.npy", stats)
    print(f"Packed {n} chips into {PACKED.resolve()}")
    print(f"Labelled pixels: {(labels != IGNORE).sum():,}")


if __name__ == "__main__":
    main()