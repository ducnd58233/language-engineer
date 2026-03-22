from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def upload(adapter_path: str, repo_id: str, private: bool) -> None:
    """Upload final adapter weights only (no optimizer/scheduler state)."""
    path = Path(adapter_path)
    if (path / "final").is_dir():
        path = path / "final"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*"],
    )
    print(f"Uploaded {path} -> https://huggingface.co/{repo_id}")


def upload_checkpoint(checkpoint_path: str, repo_id: str, private: bool) -> None:
    """Upload full checkpoint (adapter + optimizer + scheduler + trainer state) for training resume."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="last-checkpoint",
    )
    print(
        f"Uploaded checkpoint {checkpoint_path} -> https://huggingface.co/{repo_id}/last-checkpoint"
    )


def download(repo_id: str, output_path: str) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_path,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {repo_id} -> {output_path}")


def download_checkpoint(repo_id: str, output_path: str) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_path,
        token=os.environ.get("HF_TOKEN"),
    )
    checkpoint_dir = Path(output_path) / "last-checkpoint"
    if checkpoint_dir.exists():
        print(f"Downloaded {repo_id} -> {checkpoint_dir}")
        print(
            f"Use for resume: set resume.checkpoint_path: {checkpoint_dir} in lora_config.yaml"
        )
    else:
        print(
            f"Downloaded {repo_id} -> {output_path} (no last-checkpoint/ subdir found)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload final adapter weights to HF Hub")
    up.add_argument("--adapter", required=True, help="Path to adapter directory")
    up.add_argument("--repo", required=True, help="HF repo ID (user/repo-name)")
    up.add_argument("--private", action="store_true", help="Create private repo")

    upc = sub.add_parser(
        "upload-checkpoint",
        help="Upload full checkpoint (optimizer + scheduler) to HF Hub",
    )
    upc.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g. runs/.../checkpoint-500)",
    )
    upc.add_argument(
        "--repo", required=True, help="HF repo ID for checkpoints (user/repo-name)"
    )
    upc.add_argument("--private", action="store_true", help="Create private repo")

    dl = sub.add_parser("download", help="Download adapter from HF Hub")
    dl.add_argument("--repo", required=True, help="HF repo ID (user/repo-name)")
    dl.add_argument("--output", required=True, help="Local output directory")

    dlc = sub.add_parser(
        "download-checkpoint",
        help="Download full checkpoint (optimizer + scheduler) from HF Hub for training resume",
    )
    dlc.add_argument(
        "--repo", required=True, help="HF checkpoint repo ID (user/repo-name-ckpt)"
    )
    dlc.add_argument("--output", required=True, help="Local output directory")

    args = parser.parse_args()
    if args.command == "upload":
        upload(args.adapter, args.repo, args.private)
    elif args.command == "upload-checkpoint":
        upload_checkpoint(args.checkpoint, args.repo, args.private)
    elif args.command == "download-checkpoint":
        download_checkpoint(args.repo, args.output)
    else:
        download(args.repo, args.output)
