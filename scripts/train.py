from __future__ import annotations

import math
import os
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")  # 5 min
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / ".hf-cache"),
)
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(Path(__file__).resolve().parents[1] / ".hf-cache" / "datasets"),
)
os.environ.setdefault(
    "TMPDIR", str(Path(__file__).resolve().parents[1] / ".hf-cache" / "tmp")
)

sys.path.insert(0, str(Path(__file__).parent))

import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, utils
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from utils import build_bnb_config, format_example, load_config, load_tokenizer, run_dir

from datasets import load_dataset

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

utils.enable_progress_bars()


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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "lora_config.yaml")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Override base model (HF repo ID)")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit training to this many rows per epoch (overrides config; default: all rows)",
    )
    parser.add_argument(
        "--shard-offset",
        type=int,
        default=None,
        help="Skip first N shards before loading (overrides config; default: 0)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help="Explicit shard indices to train on in any order (e.g. --shards 1 3 2); overrides --shard-offset",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Skip first N rows within the loaded stream (overrides config; default: 0)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit validation to this many rows per eval (overrides config; default: all rows)",
    )
    parser.add_argument(
        "--val-shard-offset",
        type=int,
        default=None,
        help="Skip first N val shards before loading (overrides config; default: 0)",
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        nargs="+",
        default=None,
        help="Explicit val shard indices (e.g. --val-shards 0 1); overrides --val-shard-offset",
    )
    parser.add_argument(
        "--val-offset",
        type=int,
        default=None,
        help="Skip first N rows within the val stream (overrides config; default: 0)",
    )
    args = parser.parse_args()

    model_name = args.model or cfg["model"]["base_model"]
    t = cfg["training"]
    d = cfg["data"]

    max_rows = args.max_rows or d.get("max_rows") or None
    shard_offset = (
        args.shard_offset
        if args.shard_offset is not None
        else (d.get("shard_offset") or 0)
    )
    shards = args.shards or d.get("shards") or None
    offset = args.offset if args.offset is not None else (d.get("offset") or 0)

    max_eval_samples = (
        args.max_eval_samples
        if args.max_eval_samples is not None
        else (d.get("max_eval_samples") or None)
    )
    val_shard_offset = (
        args.val_shard_offset
        if args.val_shard_offset is not None
        else (d.get("val_shard_offset") or 0)
    )
    val_shards_explicit = args.val_shards or d.get("val_shards") or None
    val_offset = (
        args.val_offset if args.val_offset is not None else (d.get("val_offset") or 0)
    )

    if args.epochs:
        t["num_train_epochs"] = args.epochs

    push_to_hub = bool(t.get("push_to_hub", False))
    hub_strategy = str(t.get("hub_strategy", "every_save"))
    hub_private_repo = bool(t.get("hub_private_repo", False))
    hub_model_id = t.get("hub_model_id") or os.environ.get("HF_REPO_ID")
    hub_token = os.environ.get("HF_TOKEN")
    hub_token_arg = hub_token if push_to_hub else None
    hub_model_id_arg = str(hub_model_id) if (push_to_hub and hub_model_id) else None
    hub_checkpoint_repo_id = f"{hub_model_id}-ckpt" if hub_model_id else None

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
        api.create_repo(
            repo_id=str(hub_checkpoint_repo_id),
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
    if shards is not None:
        train_shards = [train_shards[i] for i in shards if i < len(train_shards)]
    elif shard_offset:
        train_shards = train_shards[shard_offset:]
    val_shards = sorted(val_dir.glob("*.parquet"))
    if val_shards_explicit is not None:
        val_shards = [val_shards[i] for i in val_shards_explicit if i < len(val_shards)]
    elif val_shard_offset:
        val_shards = val_shards[val_shard_offset:]
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
    if offset:
        train_ds = train_ds.skip(offset)
    if max_rows:
        train_ds = train_ds.take(max_rows)

    _fmt = partial(
        format_example,
        tokenizer=tokenizer,
        max_seq_length=cfg["model"]["max_seq_length"],
    )

    train_ds = train_ds.shuffle(buffer_size=10_000, seed=42).repeat(
        t["num_train_epochs"]
    )
    train_ds = train_ds.map(_fmt, remove_columns=["document", "summary", "source"])

    val_ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in val_shards],
        split="train",
        streaming=True,
    )
    if val_offset:
        val_ds = val_ds.skip(val_offset)
    if max_eval_samples:
        val_ds = val_ds.take(max_eval_samples)
    val_ds = val_ds.map(_fmt, remove_columns=["document", "summary", "source"])
    if max_eval_samples:
        from datasets import Dataset as HFDataset

        val_ds = HFDataset.from_list(list(val_ds))

    num_rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in train_shards)
    if offset:
        num_rows = max(0, num_rows - offset)
    if max_rows:
        num_rows = min(num_rows, max_rows)
    print(f"Train shards: {len(train_shards)}, val shards: {len(val_shards)}")
    steps_per_epoch = math.ceil(
        num_rows / (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"])
    )
    max_steps = steps_per_epoch * t["num_train_epochs"]
    print(
        f"Training: {steps_per_epoch} steps/epoch x {t['num_train_epochs']} epochs = {max_steps} steps"
    )

    warmup_steps = max(1, int(t.get("warmup_ratio", 0.03) * max_steps))

    sft_cfg = SFTConfig(
        output_dir=str(repo_root / run_dir(model_name)),
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
        hub_model_id=hub_checkpoint_repo_id,
        hub_private_repo=hub_private_repo,
        hub_token=hub_token_arg,
        gradient_checkpointing_kwargs=t.get("gradient_checkpointing_kwargs", {}),
        optim=t.get("optim", "paged_adamw_8bit"),
    )
    resume_cfg = t.get("resume", {})
    if resume_cfg.get("enabled"):
        checkpoint_path = resume_cfg.get("checkpoint_path") or get_latest_checkpoint(
            repo_root / run_dir(model_name)
        )
        if checkpoint_path:
            sft_cfg.resume_from_checkpoint = checkpoint_path
            print(f"Resuming from local checkpoint: {checkpoint_path}")
            state_file = Path(checkpoint_path) / "trainer_state.json"
            if state_file.exists():
                import json as _json

                global_step = _json.loads(state_file.read_text())["global_step"]
                sft_cfg.max_steps = global_step + max_steps
                print(
                    f"Adjusted max_steps: {global_step} (checkpoint) + {max_steps} (segment) = {sft_cfg.max_steps}"
                )
        elif push_to_hub and hub_checkpoint_repo_id:
            from huggingface_hub import snapshot_download

            print(
                f"No local checkpoint — downloading from Hub: {hub_checkpoint_repo_id}"
            )
            hub_local = repo_root / run_dir(model_name) / "hub_checkpoint"
            snapshot_download(
                repo_id=hub_checkpoint_repo_id,
                local_dir=str(hub_local),
                token=hub_token,
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

    output_dir = repo_root / run_dir(model_name) / "final"
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")

    if push_to_hub:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=str(hub_model_id_arg),
            repo_type="model",
        )
        print(f"Final adapter pushed → {hub_model_id_arg}")


if __name__ == "__main__":
    main()
