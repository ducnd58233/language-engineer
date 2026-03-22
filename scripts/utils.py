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

CHUNK_PROMPT = (
    "From the following passage, extract only specific facts, discoveries, events, "
    "and named entities that are unique to this section. "
    "Skip any repeated descriptions, standard setting details, or boilerplate phrasing. "
    "Be concise and factual.\n\n"
    "Passage:\n{document}\n\nKey facts:\n"
)

REFINE_PROMPT = (
    "You have the following summary so far:\n{existing_summary}\n\n"
    "New section:\n{document}\n\n"
    "Update the summary to incorporate any new information from this section. "
    "Keep all previously captured facts. Add new entities, events, and discoveries. "
    "Avoid repeating information already in the summary.\n\nUpdated summary:\n"
)

REFINE_TOKENS = 1024
MIN_SUMMARY_TOKENS = 512


def format_example(example: dict, tokenizer=None, max_seq_length: int = 0) -> dict:
    document = example["document"]
    if tokenizer and max_seq_length:
        prompt_len = len(
            tokenizer(PROMPT.format(document=""), add_special_tokens=False)["input_ids"]
        )
        summary_len = len(
            tokenizer(example["summary"], add_special_tokens=False)["input_ids"]
        )
        doc_budget = (
            max_seq_length - prompt_len - summary_len - 2
        )  # 2 for eos and pad tokens
        doc_ids = tokenizer(document, add_special_tokens=False)["input_ids"]
        if 0 < doc_budget < len(doc_ids):
            document = tokenizer.decode(doc_ids[:doc_budget], skip_special_tokens=True)
    example["text"] = PROMPT.format(document=document) + example["summary"]
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


def run_dir(model_name: str) -> str:
    return "runs/" + model_name.split("/")[-1] + "_qlora"


def chunk_document(
    tokenizer, document: str, chunk_size: int, overlap: int
) -> list[str]:
    doc_tokens = tokenizer(document, add_special_tokens=False)["input_ids"]
    step = max(1, chunk_size - overlap)
    return [
        tokenizer.decode(doc_tokens[i : i + chunk_size], skip_special_tokens=True)
        for i in range(0, len(doc_tokens), step)
    ]


def fits_in_context(
    tokenizer, document: str, model_ctx: int, reserve_tokens: int
) -> bool:
    prompt_overhead = len(
        tokenizer(PROMPT.format(document=""), add_special_tokens=False)["input_ids"]
    )
    doc_len = len(tokenizer(document, add_special_tokens=False)["input_ids"])
    return doc_len <= model_ctx - prompt_overhead - reserve_tokens


def generate_summary(
    model,
    tokenizer,
    document: str,
    max_new_tokens: int = MIN_SUMMARY_TOKENS,
    prompt_template: str = PROMPT,
) -> str:
    prompt = prompt_template.format(document=document)
    max_input = model.config.max_position_embeddings - max_new_tokens
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input
    ).to(model.device)
    tokens_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            prompt_lookup_num_tokens=10,
        )
    result = tokenizer.decode(output[0][tokens_in:], skip_special_tokens=True).strip()
    del inputs, output
    torch.cuda.empty_cache()
    return result
