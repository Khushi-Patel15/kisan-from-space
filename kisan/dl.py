"""Deep-learning building blocks shared by the training and evaluation scripts."""
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset

from kisan.common import CLASS_IDS, DATA, IGNORE, PACKED


def load_packed():
    """Memory-mapped arrays from step 6 (fast, uses little RAM)."""
    images = np.load(PACKED / "images.npy", mmap_mode="r")
    labels = np.load(PACKED / "labels.npy")
    fields = np.load(PACKED / "fields.npy")
    chips = (PACKED / "chips.txt").read_text().split("\n")
    stats = np.load(PACKED / "band_stats.npy")
    return images, labels, fields, chips, stats


def field_splits():
    """field_id -> 'train' / 'val' / 'test' (from step 4)."""
    s = pd.read_csv(DATA / "splits.csv")
    return dict(zip(s["field_id"], s["split"]))


def split_masks(labels, fields, splits, split):
    """Copy of `labels` where only pixels of fields in `split` keep their class; rest = IGNORE.
    This is how we make sure the model NEVER sees the labels of validation/test fields."""
    ids = np.array([f for f, s in splits.items() if s == split])
    keep = np.isin(fields, ids)
    out = np.full_like(labels, IGNORE)
    out[keep] = labels[keep]
    return out


class ChipDataset(Dataset):
    """Yields (normalised 12-band image, label mask) pairs, with random crop/flip/rotate for training."""

    def __init__(self, images, masks, indices, stats, crop=None, augment=False):
        self.images, self.masks, self.idx = images, masks, np.asarray(indices)
        self.mean = stats[0][:, None, None].astype(np.float32)
        self.std = stats[1][:, None, None].astype(np.float32)
        self.crop, self.augment = crop, augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        x = (self.images[i].astype(np.float32) - self.mean) / self.std
        y = self.masks[i]
        if self.crop:  # random crop that is centred near a labelled pixel, so crops aren't empty
            ys, xs = np.nonzero(y != IGNORE)
            j = np.random.randint(len(ys))
            h = self.crop // 2
            cy = np.clip(ys[j] + np.random.randint(-h // 2, h // 2 + 1), h, 256 - h)
            cx = np.clip(xs[j] + np.random.randint(-h // 2, h // 2 + 1), h, 256 - h)
            x, y = x[:, cy - h:cy + h, cx - h:cx + h], y[cy - h:cy + h, cx - h:cx + h]
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            r = np.random.randint(4)
            x, y = np.rot90(x, r, axes=(1, 2)), np.rot90(y, r)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))


def build_model(encoder="resnet34", pretrained=True):
    """U-Net: an encoder (ResNet) that understands the image + a decoder that paints a class on every pixel."""
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=12, classes=len(CLASS_IDS))


@torch.no_grad()
def predict_probs(model, images, indices, stats, device, batch=16):
    """Yield (chip index, class probabilities (13, 256, 256)) for each chip in `indices`."""
    model.eval()
    ds = ChipDataset(images, np.zeros((len(images), 1, 1), np.uint8), indices, stats)
    for start in range(0, len(indices), batch):
        idx = indices[start:start + batch]
        x = torch.stack([ds[k][0] for k in range(start, start + len(idx))]).to(device)
        # test-time augmentation: average the prediction with a horizontally flipped copy
        p = torch.softmax(model(x), 1) + torch.softmax(model(x.flip(-1)), 1).flip(-1)
        for i, pi in zip(idx, (p / 2).cpu().numpy()):
            yield i, pi


def field_probabilities(model, images, fields, indices, stats, device, wanted=None):
    """Average the pixel probabilities over each field -> {field_id: (13,) probability vector}.
    A field's crop = the class with the highest average probability over its pixels."""
    sums, counts = {}, {}
    for i, probs in predict_probs(model, images, indices, stats, device):
        fid = fields[i]
        for f in np.unique(fid[fid > 0]):
            if wanted is not None and f not in wanted:
                continue
            m = fid == f
            sums[f] = sums.get(f, 0) + probs[:, m].sum(1)
            counts[f] = counts.get(f, 0) + m.sum()
    return {f: sums[f] / counts[f] for f in sums}