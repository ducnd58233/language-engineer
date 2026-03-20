from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from utils import (
    build_bnb_config,
    chunk_and_summarize,
    load_config,
    load_model,
    load_tokenizer,
)

from datasets import load_dataset


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    return bleu.corpus_score(predictions, [references]).score


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def evaluate(
    model, tokenizer, documents: list[str], references: list[str], label: str
) -> dict:
    print(f"\n[{label}] Generating {len(documents)} summaries...")
    predictions = []
    for i, doc in enumerate(documents):
        predictions.append(chunk_and_summarize(model, tokenizer, doc))
        print(f"  {i + 1}/{len(documents)}", end="\r")
    print()

    rouge = compute_rouge(predictions, references)
    bleu = compute_bleu(predictions, references)
    bs = compute_bertscore(predictions, references)

    return {
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bleu4": round(bleu, 2),
        "bertscore_f1": round(bs, 4),
    }


def print_table(results: dict) -> None:
    print("\n" + "=" * 52)
    print(f"{'Metric':<16} {'Base':>10} {'Adapter':>10} {'Delta':>10}")
    print("-" * 52)
    for k in results["base"]:
        b = results["base"][k]
        a = results["adapter"][k]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"{k:<16} {b:>10.4f} {a:>10.4f} {sign}{delta:>9.4f}")
    print("=" * 52)


def _run_dir(model_name: str) -> str:
    return "runs/" + model_name.split("/")[-1] + "_qlora"


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")
    model_name = args.model or cfg["model"]["base_model"]
    bnb_cfg = build_bnb_config(cfg)

    test_path = repo_root / cfg["data"]["test_parquet"]
    print(f"Loading test data from {test_path}")
    test_ds = load_dataset("parquet", data_files=str(test_path), split="train")
    test_ds = test_ds.select(range(min(args.n_samples, len(test_ds))))
    documents = test_ds["document"]
    references = test_ds["summary"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    results = {}

    print("\n--- Evaluating base model (zero-shot) ---")
    base_model = load_model(model_name, bnb_cfg, adapter_path=None)
    results["base"] = evaluate(
        base_model, tokenizer, documents, references, label="base"
    )
    del base_model
    torch.cuda.empty_cache()

    adapter_path = args.adapter or str(repo_root / _run_dir(model_name) / "final")
    print(f"\n--- Evaluating adapter: {adapter_path} ---")
    adapter_model = load_model(model_name, bnb_cfg, adapter_path=adapter_path)
    results["adapter"] = evaluate(
        adapter_model, tokenizer, documents, references, label="adapter"
    )

    print_table(results)

    out_path = repo_root / "results" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    main(parser.parse_args())
