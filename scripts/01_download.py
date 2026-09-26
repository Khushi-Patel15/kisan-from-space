"""Step 1: download the AgriFieldNet training data (~350 MB).

For every image tile ("chip") that has training labels, downloads:
  - 12 Sentinel-2 bands   -> data/chips/<chip_id>/B01.tif ... B12.tif
  - crop label raster     -> data/chips/<chip_id>/label.tif     (pixel value = crop ID, 0 = unlabelled)
  - field ID raster       -> data/chips/<chip_id>/field_ids.tif (pixel value = field ID, 0 = not a field)
Also downloads Indian state boundaries -> data/india_states.geojson

Safe to re-run: files already on disk are skipped.
Run from the project root:  python scripts/01_download.py
"""
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import BANDS, CHIPS, DATA  # noqa: E402

BASE = "https://data.source.coop/radiantearth/agrifieldnet-competition/"
LABEL_PREFIX = "train_labels/ref_agrifieldnet_competition_v1_labels_train_"
SOURCE = "source/ref_agrifieldnet_competition_v1_source_{c}/ref_agrifieldnet_competition_v1_source_{c}_{b}_10m.tif"
STATES_URL = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/"
              "releaseData/gbOpen/IND/ADM1/geoBoundaries-IND-ADM1_simplified.geojson")


def list_training_chips():
    """Find every chip ID that has training labels.

    The bucket listing returns at most 1000 names per request, so we ask once per
    first hex character of the chip ID (0-9, a-f); each group is well under 1000.
    """
    chip_ids = set()
    for h in "0123456789abcdef":
        xml = requests.get(BASE, params={"prefix": LABEL_PREFIX + h}, timeout=60).text
        for key in re.findall(r"<Key>([^<]*)</Key>", xml):
            chip_ids.add(key.split("_labels_train_")[1][:5])
    return sorted(chip_ids)


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):  # retry on network hiccups
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return
        except requests.RequestException:
            if attempt == 3:
                raise


def main():
    chips = list_training_chips()
    print(f"Found {len(chips)} chips with training labels")

    jobs = []
    for c in chips:
        folder = CHIPS / c
        jobs.append((BASE + LABEL_PREFIX + c + ".tif", folder / "label.tif"))
        jobs.append((BASE + LABEL_PREFIX + c + "_field_ids.tif", folder / "field_ids.tif"))
        for b in BANDS:
            jobs.append((BASE + SOURCE.format(c=c, b=b), folder / f"{b}.tif"))
    jobs.append((STATES_URL, DATA / "india_states.geojson"))

    with ThreadPoolExecutor(16) as pool:
        list(tqdm(pool.map(lambda j: download(*j), jobs), total=len(jobs), unit="file"))
    print("Done. Data is in", DATA.resolve())


if __name__ == "__main__":
    main()