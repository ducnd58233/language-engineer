# language-engineer

QLoRA fine-tuning pipeline for document summarization using Qwen2.5-3B-Instruct.

---

## Prerequisites

Install and configure the following before using this repository.

- **conda** (Miniconda >= 24 or Anaconda)
Required to create the Python environment from `environment.yml`.
Download: https://docs.conda.io/en/latest/miniconda.html

- **NVIDIA GPU with CUDA support**
Required for 4-bit NF4 quantization (bitsandbytes) and bf16 training.
Minimum NVIDIA driver version: 570 (supports CUDA 13.0).
Check your driver: `nvidia-smi`

- **git**
Required to clone the repository.

- **Hugging Face account and write-access token**
Required if `push_to_hub: true` in `configs/lora_config.yaml`, or if you need to download gated models.
Create a token at: https://huggingface.co/settings/tokens

---

## Setup

Run these steps in order from the repository root.

**1. Clone the repository**

```bash
git clone https://github.com/ducnd58233/language-engineer.git
cd language-engineer
```

**2. Create the conda environment**

```bash
conda env create -f environment.yml
```

This installs Python 3.13, uv, cudnn, cuda-toolkit=13.0, and accelerate into an environment named `lang-eng`

**3. Activate the environment**

```bash
conda activate lang-eng
```

**4. Install Python dependencies**

```bash
uv sync
```

Reads `pyproject.toml` and installs all packages including torch (from the PyTorch CUDA 13.0 index), transformers, peft, trl, bitsandbytes, datasets, and evaluation libraries

**5. Configure credentials**

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
HF_TOKEN=hf_your_token_here
HF_REPO_ID=your-username/your-repo-name
```

`HF_TOKEN` is required when `push_to_hub: true` in the config.
`HF_REPO_ID` overrides `hub_model_id` in the config if set.

---

## Running Scripts

Run scripts in this order for the full pipeline.

### Step 1 — Process Datasets

Downloads CNN/DailyMail, XSum from Hugging Face and writes unified parquet shards to `datasets/processed/`

```bash
uv run python scripts/process_datasets.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| --rows-per-shard | int | 100000 | Target number of rows per output parquet shard |
| --min-shards | int | 1 | Minimum number of shards regardless of row count |

Example:

```bash
uv run python scripts/process_datasets.py --rows-per-shard 50000
```

---

### Step 2 — Train

Fine-tunes Qwen2.5-3B-Instruct with QLoRA (4-bit NF4, LoRA r=16, alpha=32). All hyperparameters are read from `configs/lora_config.yaml`. CLI flags override config values for that run only.

```bash
uv run python scripts/train.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| --model | str | (from config) | Override base model HF repo ID |
| --epochs | int | (from config) | Override number of training epochs |
| --max-rows | int | (from config) | Limit training rows per epoch |
| --shard-offset | int | 0 | Skip first N training shards |
| --shards | int... | (from config) | Explicit shard indices to use, e.g. --shards 1 3 2 |
| --offset | int | 0 | Skip first N rows within the loaded stream |
| --max-eval-samples | int | (from config) | Limit validation rows per eval step |
| --val-shard-offset | int | 0 | Skip first N validation shards |
| --val-shards | int... | (from config) | Explicit val shard indices, e.g. --val-shards 0 1 |
| --val-offset | int | 0 | Skip first N rows within the val stream |

The trained adapter is saved to `runs/Qwen2.5-3B-Instruct_qlora/final/` after training completes.

If `push_to_hub: true` in the config, `HF_TOKEN` must be set in `.env`. The adapter is pushed to `hub_model_id` and the best checkpoint is pushed to `hub_model_id-ckpt`.

Example — train on first 5000 rows, evaluate on 100 validation samples:

```bash
uv run python scripts/train.py --max-rows 5000 --max-eval-samples 100
```

---

### Step 3 — Evaluate

Evaluates the base model (zero-shot) and the fine-tuned adapter side-by-side on the test set. Reports ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, and BERTScore F1, and prints a comparison table. Results are saved to `results/eval_results.json`.

```bash
uv run python scripts/evaluate.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| --adapter | str | runs/Qwen2.5-3B-Instruct_qlora/final | Path to the LoRA adapter directory |
| --n-samples | int | 500 | Number of test samples to evaluate |
| --model | str | (from config) | Override base model HF repo ID |
| --strategy | str | hierarchical | Summarization strategy: map-refine, hierarchical, extract-abstract, all. When `all`, results are saved incrementally after each strategy |

---

### Step 4 — Inference

Summarizes a single document. Provide either `--document` (inline text) or `--input` (path to a .txt file); these two flags are mutually exclusive. When `--strategy all` is used, pairwise cross-comparison metrics (ROUGE-1/2/L, BLEU-4, BERTScore) are computed between all strategy outputs.

```bash
uv run python scripts/inference.py [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| --document | str | - | Document text to summarize (mutually exclusive with --input) |
| --input | str | - | Path to a .txt input file (mutually exclusive with --document) |
| --output | str | results/inference_result.json (when --strategy all) | Save result to file: .txt writes summary only, .json writes full result with metadata |
| --adapter | str | runs/Qwen2.5-3B-Instruct_qlora/final | Path to the LoRA adapter directory |
| --model | str | (from config) | Override base model HF repo ID |
| --strategy | str | hierarchical | Strategy: map-refine, hierarchical, extract-abstract, all. When `all`, pairwise cross-comparison metrics are saved under `cross_comparison` in the output |

---

## Hub Utilities (optional)

Use `scripts/hub.py` to upload or download model artifacts manually. This is useful when `push_to_hub` is disabled in the config or when you want to manage artifacts independently of training.

```bash
uv run python scripts/hub.py <subcommand> [OPTIONS]
```

### upload

Upload final adapter weights to a HF Hub repository. Accepts either the run root (`runs/Qwen2.5-3B-Instruct_qlora/`) or the `final/` subdirectory directly — auto-resolves to `final/` if present. Checkpoint directories are excluded automatically.

```bash
uv run python scripts/hub.py upload --adapter <path> --repo <repo-id> [--private]
```

| Argument | Required | Description |
|---|---|---|
| --adapter | yes | Path to run root or final/ dir, e.g. runs/Qwen2.5-3B-Instruct_qlora/ |
| --repo | yes | HF repo ID, e.g. username/repo-name |
| --private | no | Create the repository as private |

### upload-checkpoint

Upload a full training checkpoint (including optimizer and scheduler state) to a separate checkpoint repository. Uploaded under `last-checkpoint/` in the target repo, matching the structure expected by the resume logic in `train.py`.

```bash
uv run python scripts/hub.py upload-checkpoint --checkpoint <path> --repo <repo-id> [--private]
```

| Argument | Required | Description |
|---|---|---|
| --checkpoint | yes | Path to checkpoint directory, e.g. runs/Qwen2.5-3B-Instruct_qlora/checkpoint-500 |
| --repo | yes | HF checkpoint repo ID (typically hub_model_id-ckpt) |
| --private | no | Create the repository as private |

### download

Download adapter weights from a HF Hub repository to a local directory.

```bash
uv run python scripts/hub.py download --repo <repo-id> --output <local-dir>
```

| Argument | Required | Description |
|---|---|---|
| --repo | yes | HF repo ID to download from |
| --output | yes | Local directory to save the downloaded files |

### download-checkpoint

Download a full training checkpoint from the checkpoint repository for training resume. Prints the exact path to set as `resume.checkpoint_path` in `lora_config.yaml`.

```bash
uv run python scripts/hub.py download-checkpoint --repo <ckpt-repo-id> --output <local-dir>
```

| Argument | Required | Description |
|---|---|---|
| --repo | yes | HF checkpoint repo ID (typically hub_model_id-ckpt) |
| --output | yes | Local directory, e.g. runs/Qwen2.5-3B-Instruct_qlora/hub_checkpoint |

Example:

```bash
uv run python scripts/hub.py download-checkpoint \
  --repo username/repo-name-ckpt \
  --output runs/Qwen2.5-3B-Instruct_qlora/hub_checkpoint
# Output: Use for resume: set resume.checkpoint_path: runs/.../hub_checkpoint/last-checkpoint
```

---

## Configuration

All hyperparameters are in `configs/lora_config.yaml`. Edit this file to change training behavior without touching script code.

Key sections:

- `model` — base model ID and max sequence length
- `quantization` — 4-bit NF4 quantization settings (bitsandbytes)
- `lora` — LoRA rank, alpha, dropout, and target projection modules
- `training` — batch size, learning rate, scheduler, eval/save strategy, hub push settings
- `data` — dataset directory paths and row/shard windowing controls

---

## Output Layout

| Path | Contents |
|---|---|
| datasets/processed/train/ | Preprocessed training parquet shards |
| datasets/processed/validation/ | Preprocessed validation parquet shards |
| datasets/processed/test/ | Preprocessed test parquet shards |
| runs/Qwen2.5-3B-Instruct_qlora/ | Training run directory |
| runs/Qwen2.5-3B-Instruct_qlora/checkpoint-N/ | Intermediate checkpoint (optimizer + scheduler state) |
| runs/Qwen2.5-3B-Instruct_qlora/final/ | Final LoRA adapter weights (used by evaluate and inference) |
| results/eval_results.json | Evaluation scores (ROUGE, BLEU, BERTScore) |
