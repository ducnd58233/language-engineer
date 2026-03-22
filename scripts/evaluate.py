from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / ".hf-cache"),
)

sys.path.insert(0, str(Path(__file__).parent))

import torch
from metrics import compute_all
from strategies import STRATEGIES, summarize
from tqdm import tqdm
from utils import build_bnb_config, load_config, load_model, load_tokenizer, run_dir

from datasets import load_dataset


def evaluate(
    model,
    tokenizer,
    documents: list[str],
    references: list[str],
    label: str,
    strategy: str = "hierarchical",
    max_seq_length: int = 1024,
) -> dict:
    t0 = time.time()
    predictions = []
    for doc in tqdm(documents, desc=f"[{label}/{strategy}] summarizing", unit="doc"):
        predictions.append(
            summarize(
                model, tokenizer, doc, strategy=strategy, max_seq_length=max_seq_length
            )
        )
    duration = round(time.time() - t0, 1)

    result = compute_all(predictions, references)
    result["duration_sec"] = duration
    return result


def print_table(results: dict) -> None:
    print("\n" + "=" * 76)
    print(f"{'Metric':<16} {'Base':>10} {'Adapter':>10} {'Delta':>10}  {'Winner':<8}")
    print("-" * 76)
    for k in results["base"]:
        b = results["base"][k]
        a = results["adapter"][k]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        winner = "adapter" if delta > 0 else ("base" if delta < 0 else "tie")
        print(f"{k:<16} {b:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}  {winner:<8}")
    print("=" * 76)


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")
    model_name = args.model or cfg["model"]["base_model"]
    max_seq_length = cfg["model"]["max_seq_length"]
    bnb_cfg = build_bnb_config(cfg)

    test_dir = repo_root / cfg["data"]["test_dir"]
    test_shards = sorted(test_dir.glob("*.parquet"))
    if not test_shards:
        raise FileNotFoundError(
            f"No parquet shards found in {test_dir}. Run process_datasets.py first."
        )
    print(f"Loading test data from {test_dir} ({len(test_shards)} shards)")
    test_ds = load_dataset(
        "parquet", data_files=[str(p) for p in test_shards], split="train"
    )
    test_ds = test_ds.select(range(min(args.n_samples, len(test_ds))))
    documents = list(test_ds["document"])
    references = list(test_ds["summary"])

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    strategies_to_run = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    base_results = {}
    print("\n--- Evaluating base model (zero-shot) ---")
    base_model = load_model(model_name, bnb_cfg, adapter_path=None)
    for name in strategies_to_run:
        base_results[name] = evaluate(
            base_model,
            tokenizer,
            documents,
            references,
            label="base",
            strategy=name,
            max_seq_length=max_seq_length,
        )
    del base_model
    torch.cuda.empty_cache()

    out_path = repo_root / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(strategies_to_run) > 1:
        with open(out_path, "w") as f:
            json.dump({"_partial_base": base_results}, f, indent=2)

    adapter_path = args.adapter or str(repo_root / run_dir(model_name) / "final")
    adapter_results = {}
    print(f"\n--- Evaluating adapter: {adapter_path} ---")
    adapter_model = load_model(model_name, bnb_cfg, adapter_path=adapter_path)
    for name in strategies_to_run:
        adapter_results[name] = evaluate(
            adapter_model,
            tokenizer,
            documents,
            references,
            label="adapter",
            strategy=name,
            max_seq_length=max_seq_length,
        )

        if len(strategies_to_run) > 1:
            partial = {}
            for done_name in adapter_results:
                partial[done_name] = {
                    "base": base_results[done_name],
                    "adapter": adapter_results[done_name],
                }
            with open(out_path, "w") as f:
                json.dump(partial, f, indent=2)

    if len(strategies_to_run) == 1:
        name = strategies_to_run[0]
        results = {"base": base_results[name], "adapter": adapter_results[name]}
        print_table(results)
    else:
        results = {}
        for name in strategies_to_run:
            results[name] = {
                "base": base_results[name],
                "adapter": adapter_results[name],
            }
            print(f"\n--- {name} ---")
            print_table(results[name])

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", default="", help="Path to LoRA adapter (runs/<model>/final)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Number of test samples to evaluate"
    )
    parser.add_argument("--model", default="", help="Override base model (HF repo ID)")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES) + ["all"],
        default="hierarchical",
        help="Summarization strategy (default: hierarchical)",
    )
    main(parser.parse_args())
