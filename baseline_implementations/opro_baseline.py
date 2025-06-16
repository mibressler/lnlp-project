from typing import List, Optional, Tuple, Dict

import pandas as pd

from utils.dataset import ClaudetteDataset
from utils.llm import get_llm_response
from utils.metrics import compute_binary_metrics, display_metrics


# --- Helper functions -------------------------------------------------------

def build_meta_prompt(exemplars: pd.DataFrame, solutions: List[Tuple[str, float]]) -> str:
    """Return a meta prompt to generate a new instruction."""
    prompt = (
        "You are optimizing an instruction for classifying contract clauses as fair"
        " or unfair. Below are some labelled examples followed by previously tried"
        " instructions and their accuracy. Suggest a new, different instruction"
        " likely to improve accuracy. Respond with only the instruction text.\n\n"
        "## Training Examples:\n"
    )
    for _, row in exemplars.iterrows():
        prompt += f"Clause: {row['text']}\nLabel: {row['label']}\n\n"
    prompt += "## Previous Instructions and Accuracy:\n"
    for inst, score in solutions:
        prompt += f"Instruction: {inst}\nScore: {score:.4f}\n\n"
    prompt += "## Task:\nNew instruction:"
    return prompt


def propose_instruction(prompt: str) -> str:
    """Call the LLM to get a new instruction."""
    resp = get_llm_response([{"role": "user", "content": prompt}])
    return resp.content.strip()


def score_instruction(instruction: str, sample_df: pd.DataFrame) -> Dict[str, float]:
    """Return metrics on a sample using the given instruction."""
    preds = []
    for text in sample_df["text"]:
        prompt = f"{instruction}\n\nClause: {text}\nLabel:"
        resp = get_llm_response([{"role": "user", "content": prompt}])
        raw = resp.content.strip()
        preds.append(1 if raw and raw[0] == "1" else 0)
    metrics = compute_binary_metrics(sample_df["label"], preds)
    return metrics


# --- Public API -------------------------------------------------------------

def run(
    dataset: Optional[ClaudetteDataset] = None,
    iterations: int = 3,
    *,
    sample_size: int = 50,
):
    """Run a very small OPRO loop and return metrics."""

    dataset = dataset or ClaudetteDataset()
    exemplars = dataset.get_dataset("train").sample(5, random_state=42)[["text", "label"]]
    val_sample = dataset.get_dataset("val").sample(20, random_state=42)[["text", "label"]]
    test_sample = (
        dataset.get_dataset("test").sample(sample_size, random_state=42)[["text", "label"]]
    )

    solutions: List[Tuple[str, float]] = [
        ("Classify if the clause is fair or unfair. Respond with 0 or 1 only.", 0.0)
    ]

    for i in range(iterations):
        meta_prompt = build_meta_prompt(exemplars, solutions[-5:])
        new_instruction = propose_instruction(meta_prompt)
        val_metrics = score_instruction(new_instruction, val_sample)
        print(f"Iteration {i + 1}: '{new_instruction}' -> accuracy {val_metrics['accuracy']:.4f}")
        solutions.append((new_instruction, val_metrics['accuracy']))

    best_instruction, best_acc = max(solutions[1:], key=lambda x: x[1])
    print(f"Best instruction: {best_instruction} (val acc {best_acc:.4f})")

    test_metrics = score_instruction(best_instruction, test_sample)
    display_metrics("OPRO baseline - Test", binary=test_metrics)
    params = f"iterations={iterations}"
    sample_n = len(test_sample)
    return {
        "binary_test": test_metrics,
        "multi_test": None,
        "params": params,
        "sample_size": sample_n,
    }
