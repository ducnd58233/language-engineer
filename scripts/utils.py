from __future__ import annotations

from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = (
    "Summarize the following document concisely.\n\n"
    "Document:\n{document}\n\nSummary:\n"
)

_CHUNK_SIZE = 2048
_CHUNK_OVERLAP = 256


def format_example(example: dict) -> dict:
    example["text"] = PROMPT.format(document=example["document"]) + example["summary"]
    return example


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name: str,
    bnb_cfg: BitsAndBytesConfig,
    adapter_path: str | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate_summary(model, tokenizer, document: str, max_new_tokens: int = 150) -> str:
    prompt = PROMPT.format(document=document)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1900
    ).to(model.device)
    tokens_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][tokens_in:], skip_special_tokens=True).strip()


def chunk_and_summarize(model, tokenizer, document: str) -> str:
    max_model_tokens = tokenizer.model_max_length
    prompt_overhead = len(
        tokenizer(PROMPT.format(document=""), add_special_tokens=False)["input_ids"]
    )
    doc_tokens = tokenizer(document, add_special_tokens=False)["input_ids"]

    if len(doc_tokens) <= max_model_tokens - prompt_overhead - 50:
        return generate_summary(model, tokenizer, document)

    step = _CHUNK_SIZE - _CHUNK_OVERLAP
    chunks = [
        tokenizer.decode(doc_tokens[i : i + _CHUNK_SIZE], skip_special_tokens=True)
        for i in range(0, len(doc_tokens), step)
    ]

    partial_summaries = [generate_summary(model, tokenizer, chunk) for chunk in chunks]
    combined = " ".join(partial_summaries)
    return generate_summary(model, tokenizer, combined)
