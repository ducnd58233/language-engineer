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

from strategies import STRATEGIES, summarize
from utils import build_bnb_config, load_config, load_model, load_tokenizer, run_dir


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")
    model_name = args.model or cfg["model"]["base_model"]
    max_seq_length = cfg["model"]["max_seq_length"]

    if args.input:
        document = Path(args.input).read_text(encoding="utf-8").strip()
    elif args.document:
        document = args.document
    else:
        print("Error: provide --document or --input", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(model_name)
    adapter_path = args.adapter or str(repo_root / run_dir(model_name) / "final")
    model = load_model(model_name, build_bnb_config(cfg), adapter_path=adapter_path)

    strategies_to_run = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    if len(strategies_to_run) > 1 and not args.output:
        args.output = str(repo_root / "results" / "inference_result.json")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in strategies_to_run:
        print(f"\n--- Strategy: {name} ---", file=sys.stderr)
        tokens_in = len(tokenizer(document, add_special_tokens=False)["input_ids"])
        t0 = time.time()
        summary = summarize(
            model, tokenizer, document, strategy=name, max_seq_length=max_seq_length
        )
        duration = round(time.time() - t0, 1)
        tokens_out = len(tokenizer(summary, add_special_tokens=False)["input_ids"])

        entry = {
            "summary": summary,
            "metadata": {
                "strategy": name,
                "model": model_name,
                "adapter": adapter_path if Path(adapter_path).exists() else None,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_sec": duration,
            },
        }
        results[name] = entry

        if len(strategies_to_run) > 1 and args.output:
            Path(args.output).write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if len(results) == 1:
        output = list(results.values())[0]
    else:
        output = results
        print("\n" + "=" * 72, file=sys.stderr)
        print(f"{'Strategy':<18} {'Duration':>8}", file=sys.stderr)
        print("-" * 72, file=sys.stderr)
        for name, r in results.items():
            print(f"{name:<18} {r['metadata']['duration_sec']:>7.1f}s", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        from metrics import compute_all

        comparisons = {}
        names = list(results.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                key = f"{a} vs {b}"
                comparisons[key] = compute_all(
                    [results[a]["summary"]], [results[b]["summary"]]
                )
        output["cross_comparison"] = comparisons

        print("\n" + "=" * 72, file=sys.stderr)
        print(
            f"{'Pair':<36} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BLEU-4':>7} {'BERTSc':>7}",
            file=sys.stderr,
        )
        print("-" * 72, file=sys.stderr)
        for pair, m in comparisons.items():
            print(
                f"{pair:<36} {m['rouge1']:>8.4f} {m['rouge2']:>8.4f} {m['rougeL']:>8.4f} {m['bleu4']:>7.2f} {m['bertscore_f1']:>7.4f}",
                file=sys.stderr,
            )
        print("=" * 72, file=sys.stderr)

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".txt" and len(results) == 1:
            out_path.write_text(list(results.values())[0]["summary"], encoding="utf-8")
        else:
            out_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"Output saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--document", help="Document text to summarize")
    group.add_argument("--input", help="Path to input .txt file")
    parser.add_argument(
        "--output",
        help="Save result to file (.txt = summary only, .json = full result)",
    )
    parser.add_argument(
        "--adapter", default="", help="Path to LoRA adapter (runs/<model>_qlora/final)"
    )
    parser.add_argument("--model", default="", help="Override base model (HF repo ID)")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES) + ["all"],
        default="hierarchical",
        help="Summarization strategy (default: hierarchical)",
    )
    main(parser.parse_args())
