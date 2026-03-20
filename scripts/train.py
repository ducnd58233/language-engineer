from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(__file__).resolve().parents[1] / ".cache" / "datasets"),
)
os.environ.setdefault(
    "TMPDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "tmp")
)

sys.path.insert(0, str(Path(__file__).parent))

import pyarrow.parquet as pq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from utils import build_bnb_config, load_config, load_tokenizer

from datasets import disable_caching, load_dataset

disable_caching()


def get_latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def build_lora_config(cfg: dict) -> LoraConfig:
    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        task_type=lora["task_type"],
        target_modules=lora["target_modules"],
    )


def _run_dir(model_name: str) -> str:
    return "runs/" + model_name.split("/")[-1] + "_qlora"


def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")

    model_name = args.model or cfg["model"]["base_model"]
    t = cfg["training"]
    d = cfg["data"]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)
    tokenizer.model_max_length = cfg["model"]["max_seq_length"]

    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name} (4-bit NF4)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(cfg))
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_path = repo_root / d["train_parquet"]
    val_path = repo_root / d["validation_parquet"]

    train_ds = load_dataset(
        "parquet", data_files=str(train_path), split="train", streaming=True
    )
    train_ds = train_ds.map(lambda x: {"text": x["prompt"] + x["completion"]})
    train_ds = train_ds.shuffle(buffer_size=10_000, seed=42).repeat()

    val_ds = load_dataset(
        "parquet", data_files=str(val_path), split="train", streaming=True
    )
    val_ds = val_ds.map(lambda x: {"text": x["prompt"] + x["completion"]})

    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        num_rows = pq.ParquetFile(train_path).metadata.num_rows
        steps_per_epoch = math.ceil(
            num_rows
            / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
        )
        max_steps = steps_per_epoch * t["num_train_epochs"]
        print(
            f"Auto max_steps: {num_rows} rows → {max_steps} steps ({t['num_train_epochs']} epochs)"
        )

    sft_cfg = SFTConfig(
        output_dir=str(repo_root / _run_dir(model_name)),
        dataset_text_field="text",
        max_steps=max_steps,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        report_to=t["report_to"],
    )
    resume_cfg = t.get("resume", {})
    if resume_cfg.get("enabled"):
        checkpoint_path = resume_cfg.get("checkpoint_path") or get_latest_checkpoint(
            repo_root / _run_dir(model_name)
        )
        if checkpoint_path:
            sft_cfg.resume_from_checkpoint = checkpoint_path
        else:
            sft_cfg.resume_from_checkpoint = resume_cfg.get("auto_latest", True)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Training...")
    try:
        if sft_cfg.resume_from_checkpoint:
            print(f"Resuming from checkpoint: {sft_cfg.resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=sft_cfg.resume_from_checkpoint)
        else:
            print("Training from scratch")
            trainer.train()
    except Exception as e:
        print(f"Error resuming from checkpoint: {e}")
        trainer.train()

    output_dir = repo_root / _run_dir(model_name) / "final"
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Override max training steps (0 = auto-compute from dataset size × epochs)",
    )
    parser.add_argument("--model", default="", help="Override base model (HF repo ID)")
    main(parser.parse_args())
