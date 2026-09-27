"""Step 9: prepare ANY area near the AgriFieldNet regions for the model -> data/areas/<name>/

  1. finds the Sentinel-2 scene AgriFieldNet used for that region (same season/date)
  2. learns that scene's 8-bit conversion from nearby dataset chips
  3. downloads the 12 bands for your area and converts them
  4. downloads a cropland mask (ESA WorldCover 2021) so houses, roads, water and trees
     are not coloured as crops
Example (5 km x 5 km around Nanpara, Bahraich, UP):
  python scripts/09_prepare_area.py --name nanpara --lon 81.35 --lat 27.86 --size-km 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import DATA  # noqa: E402
from kisan.s2 import area_grid, calibrate, cropland_mask, nearest_chips, read_bands, to_uint8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="short name for the area, e.g. nanpara")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--size-km", type=float, default=5, help="side of the square area (max 20)")
    args = ap.parse_args()
    size = min(args.size_km, 20)
    folder = DATA / "areas" / args.name
    folder.mkdir(parents=True, exist_ok=True)

    print("1/4 Finding the dataset scene and learning its conversion...")
    item, coef, info = calibrate(args.lon, args.lat)
    print(f"    scene {info['scene']} ({info['date']}), match {info['scene_match']}, "
          f"calibrated from {len(info['calibration_chips'])} chips")

    crs = nearest_chips(args.lon, args.lat, 1)["crs"].iloc[0]
    bounds, transform = area_grid(args.lon, args.lat, size, crs)
    print(f"2/4 Downloading 12 bands for {size:g} km x {size:g} km...")
    refl = read_bands(item, crs, bounds)
    image = to_uint8(refl, coef)
    nodata = ~np.isfinite(refl).all(axis=0)

    print("3/4 Downloading the cropland mask (ESA WorldCover)...")
    crop = cropland_mask(crs, bounds) & ~nodata

    print("4/4 Saving...")
    profile = {"driver": "GTiff", "height": image.shape[1], "width": image.shape[2], "crs": crs,
               "transform": transform, "compress": "deflate"}
    with rasterio.open(folder / "image.tif", "w", count=12, dtype="uint8", **profile) as dst:
        dst.write(image)
    with rasterio.open(folder / "cropland.tif", "w", count=1, dtype="uint8", **profile) as dst:
        dst.write(crop.astype(np.uint8), 1)
    info.update({"name": args.name, "lon": args.lon, "lat": args.lat, "size_km": size, "crs": crs,
                 "cropland_share": round(float(crop.mean()), 3), "conversion": coef})
    (folder / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Done: {folder}  (cropland = {crop.mean():.0%} of the area)")


if __name__ == "__main__":
    main()