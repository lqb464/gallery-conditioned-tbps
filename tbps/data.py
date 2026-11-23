from __future__ import annotations

import json
from functools import lru_cache
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


@lru_cache(maxsize=4)
def _load_resolved_records(root_value: str):
    """Read annotations once; resolve canonical RSTPReid paths without per-image tree scans."""
    root = Path(root_value)
    bases = [root, root / "RSTPReid", root / "rstpreid"]
    anno = next((base / "data_captions.json" for base in bases
                 if (base / "data_captions.json").is_file()), None)
    if anno is not None:
        obj = json.loads(anno.read_text())
        rows = obj if isinstance(obj, list) else next(
            (obj[k] for k in ("annotations", "data", "images") if isinstance(obj.get(k), list)), [])
        image_roots = [anno.parent, anno.parent / "imgs", anno.parent / "images"]
    else:
        parquet = next((p for base in bases if base.is_dir() for p in base.glob("*.parquet")), None)
        if parquet is None:
            raise RuntimeError(f"RSTPReid annotation data_captions.json not found under {root}. Set DATA_ROOT to the RSTPReid folder.")
        rows = pd.read_parquet(parquet).to_dict("records")
        image_roots = [parquet.parent, parquet.parent / "imgs", parquet.parent / "images"]

    # Only build a recursive index if the common RSTPReid paths do not resolve.
    resolved = []
    missing = []
    for row in rows:
        rel = row.get("img_path", row.get("file_path", row.get("image_path", row.get("image"))))
        if isinstance(rel, dict):
            rel = rel.get("path") or rel.get("filename")
        if not rel:
            continue
        p = Path(str(rel))
        image = p if p.is_absolute() and p.is_file() else next(
            (base / p for base in image_roots if (base / p).is_file()), None)
        if image is None:
            missing.append((row, p.name))
        else:
            resolved.append((row, image))
    if missing:
        index = {p.name: p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        resolved.extend((row, index[name]) for row, name in missing if name in index)

    by_split = {"train": {}, "val": {}, "test": {}}
    for row, image in resolved:
        split = str(row.get("split", "train")).lower()
        split = "val" if split in {"val", "validation", "valid"} else split
        if split not in by_split:
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
        if not captions:
            continue
        path = str(image)
        record = {"path": path, "pid": str(row.get("id", row.get("person_id", row.get("pid", image.stem.split("_")[0])))), "captions": captions}
        if path in by_split[split]:
            old = by_split[split][path]
            old["captions"] = list(dict.fromkeys(old["captions"] + captions))
        else:
            by_split[split][path] = record
    return {split: list(records.values()) for split, records in by_split.items()}


def load_records(root: str | Path, split: str):
    split = "val" if split in {"validation", "valid"} else split
    records = _load_resolved_records(str(Path(root).resolve())).get(split, [])
    if not records:
        raise RuntimeError(f"No RSTPReid images/captions found for split={split} under {root}.")
    return records


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
