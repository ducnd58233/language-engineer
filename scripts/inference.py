from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    build_bnb_config,
    chunk_and_summarize,
    load_config,
    load_model,
    load_tokenizer,
)


def _run_dir(model_name: str) -> str:
    return "runs/" + model_name.split("/")[-1] + "_qlora"


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")
    model_name = args.model or cfg["model"]["base_model"]

    if args.input:
        document = Path(args.input).read_text(encoding="utf-8").strip()
    elif args.document:
        document = args.document
    else:
        print("Error: provide --document or --input", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(model_name)
    adapter_path = args.adapter or str(repo_root / _run_dir(model_name) / "final")
    model = load_model(model_name, build_bnb_config(cfg), adapter_path=adapter_path)

    tokens_in = len(tokenizer(document, add_special_tokens=False)["input_ids"])
    summary = chunk_and_summarize(model, tokenizer, document)
    tokens_out = len(tokenizer(summary, add_special_tokens=False)["input_ids"])

    result = {
        "summary": summary,
        "metadata": {
            "model": model_name,
            "adapter": adapter_path if Path(adapter_path).exists() else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".txt":
            out_path.write_text(summary, encoding="utf-8")
        else:
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
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
    main(parser.parse_args())
