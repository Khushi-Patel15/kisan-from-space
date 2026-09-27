"""Fetching fresh Sentinel-2 images and converting them to AgriFieldNet's 8-bit format.

What we found out about AgriFieldNet's images (see the Week 3 notes):
  - each chip is a real Sentinel-2 L2A scene, e.g. Uttar Pradesh = 28 April 2022
  - reflectance was turned into 0-255 values with a LINEAR stretch that differs per scene
  - 20 m / 60 m bands were resampled to 10 m with nearest-neighbour
So to prepare a new area we: find the scene AgriFieldNet used nearby, learn that scene's
stretch (slope + intercept per band) from nearby dataset chips, and apply it to the new area.
Data source: Earth Search (AWS open data), free, no account needed.
"""
import json
import os

# Speed + reliability settings for reading satellite files over the internet (must be set
# before rasterio opens any file). Without these, GDAL makes many extra requests and a single
# stuck connection can hang forever.
for key, value in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",    # don't list the remote folder on every open
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",     # only request .tif files
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",    # fewer, bigger requests
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_TIMEOUT": "60",                      # give up on a stuck request after 60 s...
    "GDAL_HTTP_MAX_RETRY": "4",                     # ...and retry it
    "GDAL_HTTP_RETRY_DELAY": "3",
}.items():
    os.environ.setdefault(key, value)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import requests  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.warp import transform, transform_bounds  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from kisan.common import BANDS, CHIPS, DATA  # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1"
ASSET = {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B06": "rededge2",
         "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09", "B11": "swir16", "B12": "swir22"}
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


# ---------- where are the dataset chips? ----------
def chip_index():
    """Table of every dataset chip: id, CRS, bounds, centre lon/lat. Cached in data/chip_index.csv."""
    path = DATA / "chip_index.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for folder in sorted(p for p in CHIPS.iterdir() if p.is_dir()):
        with rasterio.open(folder / "B04.tif") as src:
            b, crs = src.bounds, src.crs
        lon, lat = transform(crs, "EPSG:4326", [(b.left + b.right) / 2], [(b.top + b.bottom) / 2])
        rows.append({"chip": folder.name, "crs": str(crs), "left": b.left, "bottom": b.bottom,
                     "right": b.right, "top": b.top, "lon": lon[0], "lat": lat[0]})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def nearest_chips(lon, lat, n=6):
    df = chip_index()
    d = np.hypot((df["lon"] - lon) * np.cos(np.radians(lat)), df["lat"] - lat) * 111  # km
    return df.assign(dist_km=d).nsmallest(n, "dist_km")


# ---------- Sentinel-2 ----------
def search(bbox, start, end, max_cloud=30):
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": 200, "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    r = requests.post(f"{STAC}/search", json=body, timeout=60)
    r.raise_for_status()
    return sorted(r.json()["features"], key=lambda f: f["properties"]["datetime"])


def read_bands(item, crs, bounds, bands=BANDS):
    """Reflectance (len(bands), H, W) on the 10 m grid `bounds` in `crs`, nearest-neighbour resampling."""
    left, bottom, right, top = bounds
    w, h = int(round((right - left) / 10)), int(round((top - bottom) / 10))
    out = np.zeros((len(bands), h, w), dtype=np.float32)
    for i, b in enumerate(bands):
        asset = item["assets"][ASSET[b]]
        meta = asset.get("raster:bands", [{}])[0]
        with rasterio.open(asset["href"]) as src:
            if str(src.crs) == str(crs):
                win = from_bounds(left, bottom, right, top, transform=src.transform)
                a = src.read(1, window=win, out_shape=(h, w), resampling=Resampling.nearest, boundless=True, fill_value=0)
            else:  # area in a neighbouring UTM zone: reproject straight onto our 10 m grid
                with WarpedVRT(src, crs=crs, transform=from_origin(left, top, 10, 10), width=w, height=h,
                               resampling=Resampling.nearest, nodata=0) as vrt:
                    a = vrt.read(1)
        valid = a > 0
        out[i] = np.where(valid, a * meta.get("scale", 1e-4) + meta.get("offset", 0), np.nan)
    return out


def read_chip_band(chip, band):
    with rasterio.open(CHIPS / chip / f"{band}.tif") as src:
        return src.read(1).astype(np.float32)


def find_dataset_scene(chip_row, start="2022-02-01", end="2022-06-30"):
    """The Sentinel-2 scene AgriFieldNet used for this chip = the one whose red band correlates best.
    The answer is remembered in data/scene_cache.json, so each chip is only searched once."""
    cache_path = DATA / "scene_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if chip_row["chip"] in cache:
        corr, item_id = cache[chip_row["chip"]]
        r = requests.get(f"{STAC}/collections/sentinel-2-l2a/items/{item_id}", timeout=60)
        r.raise_for_status()
        return corr, r.json()
    corr, item = _search_dataset_scene(chip_row, start, end)
    if item is not None:
        cache[chip_row["chip"]] = [corr, item["id"]]
        cache_path.write_text(json.dumps(cache, indent=1))
    return corr, item


def _search_dataset_scene(chip_row, start, end):
    crs = chip_row["crs"]
    bounds = (chip_row["left"], chip_row["bottom"], chip_row["right"], chip_row["top"])
    ref = read_chip_band(chip_row["chip"], "B04").ravel()
    best = (-1.0, None)
    for item in search(transform_bounds(crs, "EPSG:4326", *bounds), start, end, max_cloud=40):
        red = read_bands(item, crs, bounds, ["B04"])[0].ravel()
        ok = np.isfinite(red)
        if ok.mean() < 0.9:
            continue
        c = np.corrcoef(red[ok], ref[ok])[0, 1]
        if c > best[0]:
            best = (float(c), item)
        if c > 0.99:  # an almost perfect match: this is the scene, stop searching
            break
    return best


def fit_conversion(item, chip_rows):
    """Per-band linear stretch reflectance -> AgriFieldNet 8-bit, learned from dataset chips in this scene."""
    xs = {b: [] for b in BANDS}
    ys = {b: [] for b in BANDS}
    for _, r in chip_rows.iterrows():
        refl = read_bands(item, r["crs"], (r["left"], r["bottom"], r["right"], r["top"]))
        for i, b in enumerate(BANDS):
            ok = np.isfinite(refl[i])
            xs[b].append(refl[i][ok]); ys[b].append(read_chip_band(r["chip"], b)[ok])
    coef = {}
    for b in BANDS:
        x, y = np.concatenate(xs[b]), np.concatenate(ys[b])
        slope, icpt = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (slope * x + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        coef[b] = {"slope": float(slope), "intercept": float(icpt), "r2": float(r2)}
    return coef


def to_uint8(refl, coef):
    out = np.zeros(refl.shape, dtype=np.uint8)
    for i, b in enumerate(BANDS):
        v = coef[b]["slope"] * refl[i] + coef[b]["intercept"]
        out[i] = np.clip(np.round(np.nan_to_num(v, nan=0)), 0, 255)
    return out


# ---------- cropland mask ----------
def cropland_mask(crs, bounds):
    """True where ESA WorldCover 2021 says 'cropland' (class 40), on the same 10 m grid."""
    left, bottom, right, top = bounds
    w, h = int(round((right - left) / 10)), int(round((top - bottom) / 10))
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", *bounds)
    mask = np.zeros((h, w), dtype=bool)
    tiles = {(int(np.floor(la / 3) * 3), int(np.floor(lo / 3) * 3)) for la in (lat0, lat1) for lo in (lon0, lon1)}
    for la, lo in tiles:
        tile = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
        with rasterio.open(WORLDCOVER.format(tile=tile)) as src:
            with WarpedVRT(src, crs=crs, transform=from_origin(left, top, 10, 10), width=w, height=h,
                           resampling=Resampling.nearest, nodata=0) as vrt:
                a = vrt.read(1)
        mask |= a == 40
    return mask


def area_grid(lon, lat, size_km, crs):
    """Square area centred on lon/lat, snapped to the 10 m Sentinel-2 grid. Returns bounds and transform."""
    x, y = transform("EPSG:4326", crs, [lon], [lat])
    half = size_km * 500
    left, top = np.floor((x[0] - half) / 10) * 10, np.floor((y[0] + half) / 10) * 10
    right, bottom = left + size_km * 1000, top - size_km * 1000
    return (left, bottom, right, top), from_origin(left, top, 10, 10)


def calibrate(lon, lat, exclude=(), n_near=10, max_chips=5, min_corr=0.95):
    """Find AgriFieldNet's scene for this location and learn its 8-bit conversion.

    Returns (scene item, conversion coefficients, info dict). Raises if no dataset chips are
    close enough: the model is only valid in the regions/season it was trained on.
    """
    near = nearest_chips(lon, lat, n_near + len(exclude))
    near = near[~near["chip"].isin(exclude)].head(n_near)
    if near["dist_km"].iloc[0] > 50:
        raise ValueError(f"Nearest dataset chip is {near['dist_km'].iloc[0]:.0f} km away. Pick an area within "
                         "~50 km of the AgriFieldNet regions (the model only knows that region and season).")
    corr, item = find_dataset_scene(near.iloc[0])
    if item is None or corr < min_corr:
        raise ValueError(f"Could not identify the dataset scene near this area (best match {corr:.3f}).")
    # use nearby chips that come from the SAME scene (red band correlates almost perfectly)
    use = []
    for _, r in near.iterrows():
        red = read_bands(item, r["crs"], (r["left"], r["bottom"], r["right"], r["top"]), ["B04"])[0].ravel()
        ok = np.isfinite(red)
        if ok.mean() > 0.9 and np.corrcoef(red[ok], read_chip_band(r["chip"], "B04").ravel()[ok])[0, 1] >= min_corr:
            use.append(r)
        if len(use) == max_chips:
            break
    coef = fit_conversion(item, pd.DataFrame(use))
    info = {"scene": item["id"], "date": item["properties"]["datetime"][:10], "scene_match": round(float(corr), 4),
            "calibration_chips": [r["chip"] for r in use], "nearest_chip_km": round(float(near["dist_km"].iloc[0]), 1)}
    return item, coef, info