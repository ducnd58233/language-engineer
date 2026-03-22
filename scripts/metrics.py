from __future__ import annotations

from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {k: sum(v) / len(v) for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = BLEU(effective_order=True)
    return bleu.corpus_score(predictions, [references]).score


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def compute_all(predictions: list[str], references: list[str]) -> dict:
    rouge = compute_rouge(predictions, references)
    return {
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bleu4": round(compute_bleu(predictions, references), 2),
        "bertscore_f1": round(compute_bertscore(predictions, references), 4),
    }
