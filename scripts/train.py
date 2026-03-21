from __future__ import annotations

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
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from utils import build_bnb_config, format_example, load_config, load_tokenizer

from datasets import disable_caching, load_dataset

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Override base model (HF repo ID)")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    args = parser.parse_args()

    model_name = args.model or cfg["model"]["base_model"]
    t = cfg["training"]
    d = cfg["data"]

    if args.epochs:
        t["num_train_epochs"] = args.epochs

    push_to_hub = bool(t.get("push_to_hub", False))
    hub_strategy = str(t.get("hub_strategy", "every_save"))
    hub_private_repo = bool(t.get("hub_private_repo", False))
    hub_model_id = t.get("hub_model_id") or os.environ.get("HF_REPO_ID")
    hub_token = os.environ.get("HF_TOKEN")
    hub_token_arg = hub_token if push_to_hub else None
    hub_model_id_arg = str(hub_model_id) if (push_to_hub and hub_model_id) else None

    if push_to_hub:
        if not hub_model_id:
            raise ValueError(
                "push_to_hub is enabled but hub_model_id is missing. "
                "Set `training.hub_model_id` in configs/lora_config.yaml "
                "or `HF_REPO_ID` in the environment."
            )
        if not hub_token:
            raise ValueError(
                "push_to_hub is enabled but HF_TOKEN is not set in the "
                "environment (see .env.example)."
            )
        api = HfApi(token=hub_token)
        api.create_repo(
            repo_id=str(hub_model_id),
            repo_type="model",
            exist_ok=True,
            private=hub_private_repo,
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name} (4-bit NF4)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_bnb_config(cfg),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(cfg))
    model.config.use_cache = False
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_dir = repo_root / d["train_dir"]
    val_dir = repo_root / d["validation_dir"]

    train_shards = sorted(train_dir.glob("*.parquet"))
    val_shards = sorted(val_dir.glob("*.parquet"))
    if not train_shards:
        raise FileNotFoundError(
            f"No parquet shards found in {train_dir}. Run process_datasets.py first."
        )

    train_ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in train_shards],
        split="train",
        streaming=True,
    )
    train_ds = train_ds.shuffle(buffer_size=10_000, seed=42).repeat(
        t["num_train_epochs"]
    )
    train_ds = train_ds.map(
        format_example, remove_columns=["document", "summary", "source"]
    )

    val_ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in val_shards],
        split="train",
        streaming=True,
    )
    val_ds = val_ds.map(
        format_example, remove_columns=["document", "summary", "source"]
    )

    num_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in train_shards)
    print(f"Train shards: {len(train_shards)}, val shards: {len(val_shards)}")
    steps_per_epoch = math.ceil(
        num_rows / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
    )
    max_steps = steps_per_epoch * t["num_train_epochs"]
    print(
        f"Training: {num_rows} rows x {t['num_train_epochs']} epochs = {max_steps} steps"
    )

    warmup_steps = max(1, int(t.get("warmup_ratio", 0.03) * max_steps))

    sft_cfg = SFTConfig(
        output_dir=str(repo_root / _run_dir(model_name)),
        dataset_text_field="text",
        max_length=cfg["model"]["max_seq_length"],
        max_steps=max_steps,
        packing=t.get("packing", False),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=warmup_steps,
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
        push_to_hub=push_to_hub,
        hub_strategy=hub_strategy,
        hub_model_id=hub_model_id_arg,
        hub_private_repo=hub_private_repo,
        hub_token=hub_token_arg,
        gradient_checkpointing_kwargs=t.get("gradient_checkpointing_kwargs", {}),
        optim=t.get("optim", "paged_adamw_8bit"),
    )
    resume_cfg = t.get("resume", {})
    if resume_cfg.get("enabled"):
        checkpoint_path = resume_cfg.get("checkpoint_path") or get_latest_checkpoint(
            repo_root / _run_dir(model_name)
        )
        if checkpoint_path:
            sft_cfg.resume_from_checkpoint = checkpoint_path
            print(f"Resuming from local checkpoint: {checkpoint_path}")
        elif push_to_hub and hub_model_id:
            from huggingface_hub import snapshot_download

            print(f"No local checkpoint — downloading from Hub: {hub_model_id}")
            hub_local = repo_root / _run_dir(model_name) / "hub_checkpoint"
            snapshot_download(
                repo_id=hub_model_id,
                local_dir=str(hub_local),
                token=os.environ.get("HF_TOKEN"),
            )
            checkpoint_subdir = hub_local / "last-checkpoint"
            if checkpoint_subdir.exists():
                sft_cfg.resume_from_checkpoint = str(checkpoint_subdir)
                print(f"Resuming from Hub checkpoint: {checkpoint_subdir}")
            else:
                print("Hub repo has no last-checkpoint/ — starting from scratch")

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("Training...")
    if sft_cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {sft_cfg.resume_from_checkpoint}")
    else:
        print("Training from scratch")
    trainer.train(resume_from_checkpoint=sft_cfg.resume_from_checkpoint or False)

    output_dir = repo_root / _run_dir(model_name) / "final"
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")

    if push_to_hub:
        print("Pushing final adapter to HuggingFace Hub...")
        commit_info = trainer.push_to_hub(
            commit_message=(
                "Final QLoRA adapter: Qwen2.5-3B-Instruct for English summarization "
                "(CNN/DailyMail + XSum)"
            ),
            tags=["peft", "lora", "qlora", "text-summarization", "qwen2"],
            language=["en"],
            finetuned_from=model_name,
            tasks=["summarization"],
            dataset_tags=["cnn_dailymail", "xsum"],
            dataset=["cnn_dailymail", "xsum"],
        )
        print(f"Adapter pushed → {commit_info}")


if __name__ == "__main__":
    main()
