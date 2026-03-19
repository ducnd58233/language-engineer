from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset, load_dataset

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    document_column: str
    summary_column: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="cnn",
        hf_path="cnn_dailymail",
        hf_config="3.0.0",
        document_column="article",
        summary_column="highlights",
    ),
    DatasetSpec(
        name="xsum",
        hf_path="xsum",
        hf_config=None,
        document_column="document",
        summary_column="summary",
    ),
)

SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


def _normalize_text(value: str) -> str:
    normalized = re.compile(r"\s+").sub(" ", value).strip()
    return normalized


def _load_split(*, spec: DatasetSpec, split: SplitName) -> Dataset:
    if spec.hf_config is None:
        loaded = load_dataset(spec.hf_path, split=split)
        if not isinstance(loaded, Dataset):
            raise RuntimeError(f"Expected Dataset for {spec.hf_path}:{split}")
        return loaded

    loaded = load_dataset(spec.hf_path, spec.hf_config, split=split)
    if not isinstance(loaded, Dataset):
        raise RuntimeError(
            f"Expected Dataset for {spec.hf_path}/{spec.hf_config}:{split}"
        )
    return loaded


def _to_document_summary(*, ds: Dataset, spec: DatasetSpec) -> Dataset:
    keep = [spec.document_column, spec.summary_column]
    missing = [col for col in keep if col not in ds.column_names]
    if missing:
        raise ValueError(
            f"Dataset '{spec.name}' missing columns: {missing}. "
            f"Available: {ds.column_names}"
        )

    trimmed = ds.select_columns(keep)
    renamed = trimmed.rename_columns(
        {spec.document_column: "document", spec.summary_column: "summary"}
    )

    def normalize_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        documents = [_normalize_text(v) for v in batch["document"]]
        summaries = [_normalize_text(v) for v in batch["summary"]]
        return {"document": documents, "summary": summaries}

    normalized = renamed.map(normalize_batch, batched=True)

    def is_valid(example: dict[str, str]) -> bool:
        return len(example["document"]) > 0 and len(example["summary"]) > 0

    filtered = normalized.filter(is_valid)
    return filtered


def prepare_datasets(*, repo_root: Path, limit_rows: int | None) -> None:
    cache_dir = repo_root / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")

    datasets_dir = repo_root / "datasets" / "raw"

    for split in SPLITS:
        for spec in DATASETS:
            raw = _load_split(spec=spec, split=split)
            normalized = _to_document_summary(ds=raw, spec=spec)
            if limit_rows is not None:
                normalized = normalized.select(range(min(limit_rows, len(normalized))))

            path = datasets_dir / spec.name / split / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(str(path))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download data into datasets/, normalize to "
            "(document, summary), and write per-dataset per-split parquet."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional: limit rows per dataset per split (useful for quick runs).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    prepare_datasets(repo_root=args.repo_root, limit_rows=args.limit_rows)


if __name__ == "__main__":
    main()
