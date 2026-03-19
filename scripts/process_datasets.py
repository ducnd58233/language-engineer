from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from datasets import Dataset, concatenate_datasets, load_dataset

SplitName = Literal["train", "validation", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")

PROMPT = (
    "Summarize the following document concisely.\n\n"
    "Document:\n{document}\n\nSummary:"
)


def word_count(text: str) -> int:
    return len(text.split())


def hard_filter(ds: Dataset) -> tuple[Dataset, int]:
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

    ds = ds.filter(keep, desc="hard filter")
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
    example["text"] = example["prompt"] + example["summary"]
    return example


def load_all_splits(datasets_dir: Path, limit: int | None) -> dict[SplitName, Dataset]:
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
            ds = load_dataset("parquet", data_files=str(path), split="train")
            if "source" not in ds.column_names:
                ds = ds.add_column("source", [name] * len(ds))
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            parts.append(ds)
        if parts:
            result[split] = concatenate_datasets(parts)
    return result


def process(
    repo_root: Path,
    dry_run: bool,
    limit: int | None,
) -> None:
    datasets_dir = repo_root / "datasets"
    raw_dir = datasets_dir / "raw"
    processed_dir = datasets_dir / "processed"

    splits = load_all_splits(raw_dir, limit)

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
        ds, n_hard = hard_filter(ds)
        ds, n_iqr = iqr_filter(ds)
        ds, n_dup = dedup(ds)

        print(
            f"{split_name:<12} {raw:>7,}  -{n_hard:>5,}  -{n_iqr:>5,}  -{n_dup:>5,}  {len(ds):>7,}"
        )

        if dry_run:
            continue

        ds = ds.map(add_prompt, desc="format prompts")
        path = processed_dir / split_name / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(path))
        print(f"Saved {split_name} -> {path} ({len(ds)} rows)")

    if dry_run:
        print("\n[dry-run] no files written.")
    else:
        print(f"\nOutput: {processed_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Process datasets for fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--dry-run", action="store_true", help="Report only, no writes.")
    p.add_argument("--limit-rows", type=int, default=None, metavar="N")
    args = p.parse_args()
    process(repo_root=args.repo_root, dry_run=args.dry_run, limit=args.limit_rows)


if __name__ == "__main__":
    main()
