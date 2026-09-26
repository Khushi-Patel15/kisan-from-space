"""Shared settings for the Kisan from Space project."""
from pathlib import Path

# Folders (relative to the project root, where you run the scripts from)
DATA = Path("data")
CHIPS = DATA / "chips"          # one sub-folder per 256x256 image tile ("chip")
OUT = Path("outputs")

# The 12 Sentinel-2 bands in AgriFieldNet, all resampled to 10 m
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# Crop ID -> name, exactly as in the AgriFieldNet documentation
CROPS = {
    1: "Wheat", 2: "Mustard", 3: "Lentil", 4: "Fallow", 5: "Green pea", 6: "Sugarcane",
    8: "Garlic", 9: "Maize", 13: "Gram", 14: "Coriander", 15: "Potato", 16: "Bersem", 36: "Rice",
}

# One consistent colour per crop, used in every figure
CROP_COLOURS = {
    "Wheat": "#E0B43A", "Mustard": "#F4E04D", "Lentil": "#B8662A", "Fallow": "#A39E91",
    "Green pea": "#8FD16A", "Sugarcane": "#23865A", "Garlic": "#CFC6F2", "Maize": "#F08A24",
    "Gram": "#C79C74", "Coriander": "#56B8A5", "Potato": "#8A5A44", "Bersem": "#4C9A2A", "Rice": "#6DB2E3",
}

PIXEL_AREA_HA = 0.01  # one 10 m x 10 m pixel = 100 m² = 0.01 hectare


# ---- Week 2: deep learning ----
# The network predicts a class INDEX 0..12; this maps index <-> crop ID <-> name.
CLASS_IDS = sorted(CROPS)                       # [1, 2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16, 36]
CLASS_NAMES = [CROPS[c] for c in CLASS_IDS]
IGNORE = 255                                    # "no label here" value in training masks
PACKED = DATA / "packed"                        # all chips packed into a few big .npy files
MODELS = Path("models")