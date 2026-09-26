"""Step 2: turn pixels into a table with ONE ROW PER FIELD -> data/fields.csv

For every labelled field we record:
  crop, size (pixels and hectares), location (lat/lon, Indian state), and
  the mean and standard deviation of each band and of 3 vegetation indices
  over the field's pixels. These numbers are the "features" the baseline model learns from.

A field can cross the border between two chips, so values are accumulated across chips.
Run from the project root:  python scripts/02_build_field_table.py
"""
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from shapely.geometry import Point, shape
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import BANDS, CHIPS, CROPS, DATA, PIXEL_AREA_HA  # noqa: E402

INDICES = ["NDVI", "NDWI", "NDRE"]
FEATURES = BANDS + INDICES


def read_chip(folder):
    """Return (15 feature layers, crop labels, field IDs, pixel-to-map transform, CRS) for one chip."""
    bands = []
    for b in BANDS:
        with rasterio.open(folder / f"{b}.tif") as src:
            bands.append(src.read(1).astype(np.float32))
            transform_, crs = src.transform, src.crs
    x = dict(zip(BANDS, bands))
    eps = 1e-6
    # Vegetation indices: normalised differences between bands.
    # NOTE: AgriFieldNet stores bands as 8-bit values (0-255), not true reflectance,
    # so these indices are relative measures, fine for comparing fields.
    ndvi = (x["B08"] - x["B04"]) / (x["B08"] + x["B04"] + eps)   # greenness
    ndwi = (x["B03"] - x["B08"]) / (x["B03"] + x["B08"] + eps)   # water / moisture
    ndre = (x["B8A"] - x["B05"]) / (x["B8A"] + x["B05"] + eps)   # chlorophyll (red-edge)
    stack = np.stack(bands + [ndvi, ndwi, ndre])                  # shape (15, 256, 256)
    with rasterio.open(folder / "label.tif") as src:
        label = src.read(1)
    with rasterio.open(folder / "field_ids.tif") as src:
        fid = src.read(1)
    return stack, label, fid, transform_, crs


def load_states():
    """List of (state name, polygon). Accents are stripped: 'Rājasthān' -> 'Rajasthan'."""
    with open(DATA / "india_states.geojson") as f:
        gj = json.load(f)
    states = []
    for ft in gj["features"]:
        name = unicodedata.normalize("NFKD", ft["properties"]["shapeName"]).encode("ascii", "ignore").decode()
        states.append((name, shape(ft["geometry"])))
    return states


def find_state(states, lon, lat):
    pt = Point(lon, lat)
    for name, poly in states:
        if poly.contains(pt):
            return name
    # a field exactly on a (simplified) border: take the nearest state
    return min(states, key=lambda s: s[1].distance(pt))[0]


def main():
    n = len(FEATURES)
    acc = defaultdict(lambda: {"count": 0, "sum": np.zeros(n), "sumsq": np.zeros(n),
                               "crops": defaultdict(int), "chips": set(), "lon": 0.0, "lat": 0.0})
    chips = sorted(p for p in CHIPS.iterdir() if p.is_dir())
    for folder in tqdm(chips, unit="chip"):
        stack, label, fid, tr, crs = read_chip(folder)
        for f in np.unique(fid[(fid > 0) & (label > 0)]):
            m = (fid == f) & (label > 0)
            vals = stack[:, m]                                  # (15, n_pixels)
            a = acc[int(f)]
            k = vals.shape[1]
            a["count"] += k
            a["sum"] += vals.sum(1)
            a["sumsq"] += (vals.astype(np.float64) ** 2).sum(1)
            for c, cnt in zip(*np.unique(label[m], return_counts=True)):
                a["crops"][int(c)] += int(cnt)
            a["chips"].add(folder.name)
            # field centre in lon/lat (pixel-count weighted across chips)
            rows, cols = np.nonzero(m)
            xs, ys = rasterio.transform.xy(tr, rows.mean(), cols.mean())
            lon, lat = transform(crs, "EPSG:4326", [xs], [ys])
            a["lon"] += lon[0] * k
            a["lat"] += lat[0] * k

    states = load_states()
    rows = []
    for f, a in acc.items():
        k = a["count"]
        mean = a["sum"] / k
        std = np.sqrt(np.maximum(a["sumsq"] / k - mean ** 2, 0))
        crop_id = max(a["crops"], key=a["crops"].get)
        lon, lat = a["lon"] / k, a["lat"] / k
        state = find_state(states, lon, lat)
        row = {"field_id": f, "crop_id": crop_id, "crop": CROPS[crop_id],
               "label_conflict": len(a["crops"]) > 1,
               "n_pixels": k, "area_ha": k * PIXEL_AREA_HA,
               "n_chips": len(a["chips"]), "chips": ";".join(sorted(a["chips"])),
               "lon": round(lon, 5), "lat": round(lat, 5), "state": state}
        row.update({f"mean_{name}": v for name, v in zip(FEATURES, mean)})
        row.update({f"std_{name}": v for name, v in zip(FEATURES, std)})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("field_id")
    df.to_csv(DATA / "fields.csv", index=False)
    print(f"Wrote {len(df)} fields to {DATA / 'fields.csv'}")
    print(f"Fields with conflicting crop labels: {df['label_conflict'].sum()}")
    print(f"Fields split across 2+ chips: {(df['n_chips'] > 1).sum()}")
    print("\nFields per state:\n" + df["state"].value_counts().to_string())


if __name__ == "__main__":
    main()