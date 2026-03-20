from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def upload(adapter_path: str, repo_id: str, private: bool) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=adapter_path, repo_id=repo_id, repo_type="model")
    print(f"Uploaded {adapter_path} -> https://huggingface.co/{repo_id}")


def download(repo_id: str, output_path: str) -> None:
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_path,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {repo_id} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload adapter to HF Hub")
    up.add_argument("--adapter", required=True, help="Path to adapter directory")
    up.add_argument("--repo", required=True, help="HF repo ID (user/repo-name)")
    up.add_argument("--private", action="store_true", help="Create private repo")

    dl = sub.add_parser("download", help="Download adapter from HF Hub")
    dl.add_argument("--repo", required=True, help="HF repo ID (user/repo-name)")
    dl.add_argument("--output", required=True, help="Local output directory")

    args = parser.parse_args()
    if args.command == "upload":
        upload(args.adapter, args.repo, args.private)
    else:
        download(args.repo, args.output)
