"""Step 13: put ALL 1,165 training tiles and 5,551 surveyed farms on the website -> docs/data/

  tiles/<chip>.png      AI crop map of the tile (from step 8 predictions), non-cropland removed
  tiles/<chip>.jpg      the Sentinel-2 image the AI saw for that tile
  tiles.json            where each tile is
  fields.geojson        outline of every surveyed farm: farmer's crop, AI's crop, train/val/test
Run from the project root after step 8 (and step 12):  python scripts/13_export_tiles.py
Downloads a cropland mask for every tile, so it needs internet (about 5-15 min).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.features
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform, transform_geom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import CLASS_NAMES, CROP_COLOURS, CROPS, DATA, OUT, PACKED  # noqa: E402
from kisan.s2 import chip_index, cropland_mask  # noqa: E402

SITE = Path("docs") / "data"
WEB = "EPSG:3857"
NOT_CROP = 255
COLOURS = np.array([[int(CROP_COLOURS[n][i:i + 2], 16) for i in (1, 3, 5)] for n in CLASS_NAMES], dtype=np.uint8)


def to_web(arr, src_transform, crs, fill):
    """Reproject one 256x256 band from the chip's UTM grid onto the web-map grid (nearest neighbour)."""
    h, w = arr.shape
    left, top = src_transform.c, src_transform.f
    tr, W, H = calculate_default_transform(crs, WEB, w, h, left, top - 10 * h, left + 10 * w, top)
    out = np.full((H, W), fill, dtype=arr.dtype)
    reproject(arr, out, src_transform=src_transform, src_crs=crs, dst_transform=tr, dst_crs=WEB,
                resampling=Resampling.nearest, src_nodata=fill, dst_nodata=fill)
    return out, tr


def export_tile(row, pred, image):
    """Write <chip>.png (AI crops) and <chip>.jpg (satellite) and return the tile's metadata."""
    tr = from_origin(row.left, row.top, 10, 10)
    crop = pred.copy()
    crop[~cropland_mask(row.crs, (row.left, row.bottom, row.right, row.top))] = NOT_CROP
    web, wtr = to_web(crop, tr, row.crs, NOT_CROP)
    web[web >= len(CLASS_NAMES)] = NOT_CROP   # anything that isn't a real crop class = not cropland
    rgba = np.zeros((*web.shape, 4), dtype=np.uint8)
    ok = web != NOT_CROP
    rgba[ok, :3] = COLOURS[web[ok]]
    rgba[ok, 3] = 255
    Image.fromarray(rgba, "RGBA").save(SITE / "tiles" / f"{row.chip}.png", optimize=True)

    rgb = image[[3, 2, 1]].astype(float)
    for k in range(3):
        lo, hi = np.percentile(rgb[k], [1, 99.5]); rgb[k] = np.clip((rgb[k] - lo) / (hi - lo + 1e-6), 0, 1)
    bands = [to_web((rgb[k] * 255).astype(np.uint8), tr, row.crs, 0)[0] for k in range(3)]
    Image.fromarray(np.dstack(bands)).save(SITE / "tiles" / f"{row.chip}.jpg", quality=80)

    H, W = web.shape
    x0, y0 = wtr.c, wtr.f
    lons, lats = transform(WEB, "EPSG:4326", [x0, x0 + wtr.a * W], [y0, y0 + wtr.e * H])
    return {"chip": row.chip, "b": [[round(lats[1], 6), round(lons[0], 6)], [round(lats[0], 6), round(lons[1], 6)]],
            "g": [x0, y0, wtr.a, wtr.e, W, H]}


def export_fields(fields, chips, idx):
    """Outline of every surveyed farm (vectorised from the field-ID rasters), with farmer + AI crop."""
    meta = pd.read_csv(DATA / "fields.csv").merge(pd.read_csv(DATA / "splits.csv"), on="field_id").set_index("field_id")
    preds = OUT / "week2" / "field_predictions.csv"
    ai = pd.read_csv(preds).set_index("field_id") if preds.exists() else None
    feats = []
    for i, chip in enumerate(tqdm(chips, unit="chip", desc="field outlines")):
        r = idx.loc[chip]
        fid = np.asarray(fields[i]).astype(np.int32)
        for geom, f in rasterio.features.shapes(fid, mask=fid > 0, transform=from_origin(r["left"], r["top"], 10, 10)):
            f = int(f)
            if f not in meta.index:
                continue
            g = transform_geom(r["crs"], "EPSG:4326", geom, precision=5)
            m = meta.loc[f]
            props = {"id": f, "crop": m["crop"], "ha": round(float(m["area_ha"]), 2), "split": m["split"]}
            if ai is not None and f in ai.index:
                props["ai"] = CROPS[int(ai.loc[f, "U-Net"])]
                props["conf"] = round(float(ai.loc[f, "U-Net confidence"]), 2)
            feats.append({"type": "Feature", "geometry": g, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def main():
    (SITE / "tiles").mkdir(parents=True, exist_ok=True)
    images = np.load(PACKED / "images.npy", mmap_mode="r")
    fields = np.load(PACKED / "fields.npy", mmap_mode="r")
    chips = (PACKED / "chips.txt").read_text().split("\n")
    pred = np.load(DATA / "predictions" / "pred_class.npy", mmap_mode="r")
    pred_chips = (DATA / "predictions" / "chips.txt").read_text().split("\n")
    assert pred_chips == chips, "data/predictions does not match data/packed: re-run step 8"
    idx = chip_index().set_index("chip")

    def job(i):
        row = idx.loc[chips[i]]
        row = pd.Series({**row.to_dict(), "chip": chips[i]})
        return export_tile(row, np.asarray(pred[i]), np.asarray(images[i]))

    with ThreadPoolExecutor(12) as pool:   # several downloads at once (cropland masks)
        tiles = list(tqdm(pool.map(job, range(len(chips))), total=len(chips), unit="tile", desc="tiles"))
    (SITE / "tiles.json").write_text(json.dumps(tiles))

    fc = export_fields(fields, chips, idx)
    (SITE / "fields.geojson").write_text(json.dumps(fc, separators=(",", ":")))
    print(f"{len(tiles)} tiles, {len(fc['features'])} field outlines -> {SITE.resolve()}")


if __name__ == "__main__":
    main()