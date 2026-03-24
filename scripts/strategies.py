from __future__ import annotations

import os
from pathlib import Path

import nltk
from tqdm import tqdm
from utils import (
    PROMPT,
    REFINE_PROMPT,
    chunk_document,
    fits_in_context,
    generate_summary,
)

_NLTK_DIR = str(Path(__file__).resolve().parents[1] / ".nltk-cache")
os.environ.setdefault("NLTK_DATA", _NLTK_DIR)
nltk.data.path.insert(0, _NLTK_DIR)


def _download_nltk_data() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", download_dir=_NLTK_DIR, quiet=True)


def _chunk_params(
    model,
    tokenizer,
    max_seq_length: int,
    max_summary_tokens: int,
) -> tuple[int, int]:
    model_ctx = model.config.max_position_embeddings
    prompt_overhead = len(
        tokenizer(PROMPT.format(document=""), add_special_tokens=False)["input_ids"]
    )
    chunk_size = min(model_ctx - prompt_overhead - max_summary_tokens, max_seq_length)
    overlap = chunk_size // 4
    return chunk_size, overlap


def map_refine(
    model,
    tokenizer,
    document: str,
    max_seq_length: int,
    min_summary_tokens: int = 64,
    max_summary_tokens: int = 1024,
) -> str:
    chunk_size, overlap = _chunk_params(
        model, tokenizer, max_seq_length, max_summary_tokens
    )
    chunks = chunk_document(tokenizer, document, chunk_size, overlap)

    summary = generate_summary(
        model,
        tokenizer,
        chunks[0],
        max_new_tokens=max_summary_tokens,
        min_new_tokens=0,
    )
    for i, chunk in enumerate(
        tqdm(chunks[1:], desc="refine", unit="chunk", leave=False)
    ):
        is_last = i == len(chunks) - 2
        summary = generate_summary(
            model,
            tokenizer,
            REFINE_PROMPT.format(existing_summary=summary, document=chunk),
            max_new_tokens=max_summary_tokens,
            min_new_tokens=min_summary_tokens if is_last else 0,
            prompt_template="{document}",
        )
    return summary


def hierarchical_merge(
    model,
    tokenizer,
    document: str,
    max_seq_length: int,
    min_summary_tokens: int = 64,
    max_summary_tokens: int = 1024,
) -> str:
    chunk_size, overlap = _chunk_params(
        model, tokenizer, max_seq_length, max_summary_tokens
    )
    chunks = chunk_document(tokenizer, document, chunk_size, overlap)

    summaries = [
        generate_summary(
            model,
            tokenizer,
            chunk,
            max_new_tokens=max_summary_tokens,
            min_new_tokens=0,
        )
        for chunk in tqdm(chunks, desc="chunks", unit="chunk", leave=False)
    ]

    merge_prompt = (
        "Merge the following summaries into one coherent summary. "
        "Keep all key facts and remove redundancy.\n\n"
        "{document}\n\nMerged summary:\n"
    )
    while len(summaries) > 1:
        is_final = len(summaries) <= 2
        pairs = []
        for i in range(0, len(summaries), 2):
            if i + 1 < len(summaries):
                pairs.append(summaries[i] + "\n\n" + summaries[i + 1])
            else:
                pairs.append(summaries[i])
        summaries = [
            generate_summary(
                model,
                tokenizer,
                pair,
                max_new_tokens=max_summary_tokens,
                min_new_tokens=min_summary_tokens if is_final else 0,
                prompt_template=merge_prompt,
            )
            for pair in tqdm(pairs, desc="merge", unit="pair", leave=False)
        ]
    return summaries[0]


def extract_then_abstract(
    model,
    tokenizer,
    document: str,
    max_seq_length: int,
    min_summary_tokens: int = 64,
    max_summary_tokens: int = 1024,
) -> str:
    _download_nltk_data()
    from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(document, SumyTokenizer("english"))
    all_sentences = [s for p in parser.document.paragraphs for s in p.sentences]
    k = max(5, len(all_sentences) // 4)

    summarizer = TextRankSummarizer()
    extracted = summarizer(parser.document, k)
    extracted_text = " ".join(str(s) for s in extracted)

    if fits_in_context(
        tokenizer,
        extracted_text,
        model.config.max_position_embeddings,
        max_summary_tokens,
    ):
        return generate_summary(
            model,
            tokenizer,
            extracted_text,
            max_new_tokens=max_summary_tokens,
            min_new_tokens=min_summary_tokens,
        )
    return map_refine(
        model,
        tokenizer,
        extracted_text,
        max_seq_length,
        min_summary_tokens,
        max_summary_tokens,
    )


STRATEGIES = {
    "map-refine": map_refine,
    "hierarchical": hierarchical_merge,
    "extract-abstract": extract_then_abstract,
}


def summarize(
    model,
    tokenizer,
    document: str,
    strategy: str = "hierarchical",
    max_seq_length: int = 1024,
    min_summary_tokens: int = 64,
    max_summary_tokens: int = 1024,
) -> str:
    if fits_in_context(
        tokenizer, document, model.config.max_position_embeddings, max_summary_tokens
    ):
        return generate_summary(
            model,
            tokenizer,
            document,
            max_new_tokens=max_summary_tokens,
            min_new_tokens=min_summary_tokens,
        )
    return STRATEGIES[strategy](
        model,
        tokenizer,
        document,
        max_seq_length,
        min_summary_tokens,
        max_summary_tokens,
    )
