from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils import PROMPT

from datasets import Dataset, concatenate_datasets, load_dataset

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


def word_count(text: str) -> int:
    return len(text.split())


def hard_filter(ds: Dataset, num_proc: int = 1) -> tuple[Dataset, int]:
    before = len(ds)

    def keep(ex: dict) -> bool:
        dw = word_count(ex["document"])
        sw = word_count(ex["summary"])
        if dw < 20 or dw > 1500:
            return False
        if sw < 5 or sw > 150:
            return False
        if sw >= dw:
            return False
        return True

    ds = ds.filter(keep, desc="hard filter", num_proc=num_proc)
    return ds, before - len(ds)


def iqr_filter(ds: Dataset) -> tuple[Dataset, int]:
    before = len(ds)
    ratios = np.array(
        [
            word_count(s) / max(word_count(d), 1)
            for d, s in zip(ds["document"], ds["summary"])
        ]
    )
    q1, q3 = np.quantile(ratios, [0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return ds, 0
    keep_mask = (ratios >= q1 - 1.5 * iqr) & (ratios <= q3 + 1.5 * iqr)
    ds = ds.select(np.where(keep_mask)[0].tolist())
    return ds, before - len(ds)


def dedup(ds: Dataset) -> tuple[Dataset, int]:
    before = len(ds)
    seen: set[str] = set()
    keep = []
    for i, doc in enumerate(ds["document"]):
        h = hashlib.sha256(doc.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    ds = ds.select(keep)
    return ds, before - len(ds)


def add_prompt(example: dict) -> dict:
    example["prompt"] = PROMPT.format(document=example["document"])
    example["completion"] = example["summary"]
    example["text"] = example["prompt"] + example["summary"]
    return example


def load_all_splits(datasets_dir: Path, num_proc: int = 1) -> dict[SplitName, Dataset]:
    names = sorted(
        [
            c.name
            for c in datasets_dir.iterdir()
            if c.is_dir()
            and c.name not in {"all", "processed", "processed_tokenized"}
            and any((c / s / "data.parquet").exists() for s in SPLITS)
        ]
    )
    if not names:
        raise FileNotFoundError(
            f"No datasets found under {datasets_dir}. Run prepare_datasets.py first."
        )
    print(f"Datasets found: {names}")

    result: dict[SplitName, Dataset] = {}
    for split in SPLITS:
        parts = []
        for name in names:
            path = datasets_dir / name / split / "data.parquet"
            if not path.exists():
                continue
            ds = load_dataset(
                "parquet", data_files=str(path), split="train", num_proc=num_proc
            )
            if "source" not in ds.column_names:
                ds = ds.add_column("source", [name] * len(ds))
            parts.append(ds)
        if parts:
            result[split] = concatenate_datasets(parts)
    return result


def save_sharded(
    ds: Dataset, out_dir: Path, rows_per_shard: int, min_shards: int
) -> None:
    total = len(ds)
    n = max(math.ceil(total / rows_per_shard), min_shards)
    shard_size = math.ceil(total / n)
    for i in range(n):
        chunk = ds.select(range(i * shard_size, min((i + 1) * shard_size, total)))
        fname = out_dir / f"data-{i:05d}-of-{n:05d}.parquet"
        chunk.to_parquet(str(fname))
    print(f"Saved {out_dir.name} -> {out_dir} ({total} rows, {n} shards)")


def process(
    repo_root: Path, rows_per_shard: int = 100_000, min_shards: int = 1
) -> None:
    num_proc = os.cpu_count() or 1
    print(f"Using num_proc={num_proc}")

    datasets_dir = repo_root / "datasets"
    raw_dir = datasets_dir / "raw"
    processed_dir = datasets_dir / "processed"

    splits = load_all_splits(raw_dir, num_proc=num_proc)

    if "train" in splits:
        train_hashes = {
            hashlib.sha256(d.encode()).hexdigest() for d in splits["train"]["document"]
        }
        for name in ("validation", "test"):
            if name not in splits:
                continue
            ds = splits[name]
            keep = [
                i
                for i, d in enumerate(ds["document"])
                if hashlib.sha256(d.encode()).hexdigest() not in train_hashes
            ]
            leaked = len(ds) - len(keep)
            if leaked:
                print(f"[leakage] removed {leaked} rows from {name} found in train")
            splits[name] = ds.select(keep)

    print(
        f"\n{'Split':<12} {'Raw':>7}  {'Hard':>7}  {'IQR':>7}  {'Dedup':>7}  {'Final':>7}"
    )
    print("-" * 64)

    for split_name, ds in splits.items():
        raw = len(ds)
        ds, n_hard = hard_filter(ds, num_proc=num_proc)
        ds, n_iqr = iqr_filter(ds)
        ds, n_dup = dedup(ds)

        print(
            f"{split_name:<12} {raw:>7,}  -{n_hard:>5,}  -{n_iqr:>5,}  -{n_dup:>5,}  {len(ds):>7,}"
        )

        ds = ds.map(add_prompt, desc="format prompts", num_proc=num_proc)
        out_dir = processed_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_sharded(ds, out_dir, rows_per_shard=rows_per_shard, min_shards=min_shards)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100_000,
        help="Target number of rows per output parquet shard (default: 100000)",
    )
    parser.add_argument(
        "--min-shards",
        type=int,
        default=1,
        help="Minimum number of shards regardless of row count (default: 1)",
    )
    args = parser.parse_args()
    process(
        repo_root=Path(__file__).resolve().parents[1],
        rows_per_shard=args.rows_per_shard,
        min_shards=args.min_shards,
    )
