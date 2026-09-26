"""Step 4: split fields into train / validation / test -> data/splits.csv

Why not a random split? Neighbouring fields look alike (same soil, same sowing date,
same satellite image). If a field is in the test set and its next-door neighbour is in
the training set, the model gets an unfair hint and the score looks better than reality.

So we cut the map into ~10 km x 10 km blocks and put WHOLE BLOCKS into one split
("spatial block split"). StratifiedGroupKFold keeps crop proportions roughly equal
across splits while never splitting a block.
  fold 0 -> test (~20%),  fold 1 -> validation (~20%),  folds 2-4 -> train (~60%)
Run from the project root:  python scripts/04_make_splits.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kisan.common import DATA  # noqa: E402

BLOCK_DEG = 0.1  # 0.1 degree is about 10-11 km in northern India


def main():
    df = pd.read_csv(DATA / "fields.csv")
    df["block"] = (np.floor(df["lat"] / BLOCK_DEG).astype(int).astype(str) + "_" +
                   np.floor(df["lon"] / BLOCK_DEG).astype(int).astype(str))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df["split"] = "train"
    for fold, (_, idx) in enumerate(sgkf.split(df, df["crop"], groups=df["block"])):
        if fold == 0:
            df.iloc[idx, df.columns.get_loc("split")] = "test"
        elif fold == 1:
            df.iloc[idx, df.columns.get_loc("split")] = "val"

    df[["field_id", "block", "split"]].to_csv(DATA / "splits.csv", index=False)
    print(f"{df['block'].nunique()} spatial blocks")
    table = pd.crosstab(df["crop"], df["split"])[["train", "val", "test"]]
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())
    missing = [c for c in table.index[:-1] if (table.loc[c, ["val", "test"]] == 0).any()]
    if missing:
        print(f"\nNote: these rare crops are missing from val or test: {missing}. "
              "Their scores will be unreliable; mention this in the report.")


if __name__ == "__main__":
    main()