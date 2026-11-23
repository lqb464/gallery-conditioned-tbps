from __future__ import annotations

import json
from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(training: bool, image_size=(384, 128), augmentation=True):
    """IRRA/CLIP TBPS preprocessing at the ReID aspect ratio."""
    h, w = image_size
    if not training:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    ops = [T.Resize((h, w)), T.RandomHorizontalFlip(0.5)]
    if augmentation:
        ops += [T.Pad(10), T.RandomCrop((h, w))]
    ops += [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if augmentation:
        ops += [T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=CLIP_MEAN)]
    return T.Compose(ops)


def _records(root: Path):
    for path in root.rglob("*.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        rows = obj if isinstance(obj, list) else next(
            (obj[k] for k in ("annotations", "data", "images") if isinstance(obj, dict) and isinstance(obj.get(k), list)), None)
        if rows and isinstance(rows[0], dict) and any(k in rows[0] for k in ("img_path", "file_path", "image_path")):
            yield from rows


def load_records(root: str | Path, split: str):
    root = Path(root)
    rows = list(_records(root))
    if not rows:
        for p in root.rglob("*.parquet"):
            try:
                rows.extend(pd.read_parquet(p).to_dict("records"))
            except Exception:
                pass
    result = []
    for row in rows:
        row_split = str(row.get("split", "")).lower()
        if row_split and row_split not in {split, {"val": "validation", "validation": "val"}.get(split, split)}:
            continue
        rel = row.get("img_path", row.get("file_path", row.get("image_path", row.get("image"))))
        if isinstance(rel, dict):
            rel = rel.get("path") or rel.get("filename")
        if not rel:
            continue
        p = Path(str(rel))
        candidates = [p] if p.is_absolute() else [root / p, *root.rglob(p.name)]
        image = next((c for c in candidates if c.is_file()), None)
        if image is None:
            continue
        captions = row.get("captions", row.get("caption", row.get("texts", [])))
        if isinstance(captions, str):
            try:
                captions = json.loads(captions)
            except Exception:
                captions = [captions]
        if not isinstance(captions, list):
            captions = [str(captions)]
        captions = [str(c) for c in captions if c]
        if captions:
            result.append({"path": str(image), "pid": str(row.get("id", row.get("person_id", row.get("pid", image.stem.split("_")[0])))), "captions": captions})
    # Deduplicate repeated annotation rows without dropping caption variants.
    by_path = {}
    for row in result:
        if row["path"] in by_path:
            by_path[row["path"]]["captions"] = list(dict.fromkeys(by_path[row["path"]]["captions"] + row["captions"]))
        else:
            by_path[row["path"]] = row
    result = list(by_path.values())
    if not result:
        raise RuntimeError(f"No images/captions for split={split} under {root}. Inspect Kaggle dataset files and set DATA_ROOT to the extracted dataset directory.")
    return result


class PersonTextDataset(Dataset):
    def __init__(self, records, preprocess, tokenizer, training=False):
        self.preprocess, self.tokenizer = preprocess, tokenizer
        pids = sorted({r["pid"] for r in records})
        self.pid_to_label = {p: i for i, p in enumerate(pids)}
        if training:
            # Match the original protocol: each image-caption pair is a sample.
            self.records = [dict(path=r["path"], pid=r["pid"], captions=[c])
                            for r in records for c in r["captions"]]
        else:
            self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        image = self.preprocess(Image.open(r["path"]).convert("RGB"))
        caption = random.choice(r["captions"])
        return image, self.tokenizer([caption])[0], self.pid_to_label[r["pid"]]


class IdentityBatchSampler(Sampler[list[int]]):
    """Make P x K image groups so every training gallery has true positives."""

    def __init__(self, dataset: PersonTextDataset, batch_size: int, images_per_identity: int = 4):
        self.batch_size = batch_size
        self.k = images_per_identity
        self.p = batch_size // images_per_identity
        if self.p < 2:
            raise ValueError("batch_size must fit at least two identities")
        self.by_identity: dict[int, list[int]] = {}
        for idx, row in enumerate(dataset.records):
            label = dataset.pid_to_label[row["pid"]]
            self.by_identity.setdefault(label, []).append(idx)

    def __len__(self):
        return max(1, sum(map(len, self.by_identity.values())) // self.batch_size)

    def __iter__(self):
        identities = list(self.by_identity)
        for _ in range(len(self)):
            chosen = random.sample(identities, self.p)
            batch = []
            for pid in chosen:
                candidates = self.by_identity[pid]
                if len(candidates) >= self.k:
                    batch.extend(random.sample(candidates, self.k))
                else:
                    batch.extend(random.choices(candidates, k=self.k))
            yield batch
