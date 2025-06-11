from typing import Tuple

from utils.dataset import ClaudetteDataset
from utils.llm import get_llm_response, DEFAULT_MODEL
from utils.prompts import DETAILED_BINARY_TASK_INSTRUCTION
from utils.metrics import compute_binary_metrics

from our_approach.OPTS_main.popt.setting.meta_prompt_template import apet


class Prompt:
    def __init__(self, instruction: str):
        self.instruction = instruction

    def format(self, clause: str) -> str:
        return f"{self.instruction}\n\nClause: {clause}\nLabel:"


class ClaudetteEvaluator:
    """Evaluate a prompt on a small validation sample."""

    def __init__(self, n_samples: int = 20, model: str = DEFAULT_MODEL):
        dataset = ClaudetteDataset().get_dataset("val").sample(n_samples, random_state=42)
        self.texts = dataset["text"].tolist()
        self.labels = dataset["label"].tolist()
        self.model = model

    def _predict(self, prompt: Prompt) -> list[int]:
        preds = []
        for text in self.texts:
            messages = [
                {"role": "system", "content": DETAILED_BINARY_TASK_INSTRUCTION},
                {"role": "user", "content": prompt.format(text)},
            ]
            resp = get_llm_response(messages, model=self.model)
            raw = resp.content.strip().upper()
            preds.append(1 if raw.startswith("UNFAIR") or raw.startswith("1") else 0)
        return preds

    def score(self, prompt: Prompt) -> float:
        preds = self._predict(prompt)
        metrics = compute_binary_metrics(self.labels, preds)
        return metrics["accuracy"]


class SimpleOPTS:
    """Minimal OPTS loop using the APET reformulation template."""

    def __init__(self, evaluator: ClaudetteEvaluator, model: str = DEFAULT_MODEL):
        self.evaluator = evaluator
        self.model = model

    def _propose(self, current: Prompt) -> Prompt:
        meta = apet.meta_prompt_template.replace("<input>", current.instruction)
        messages = [
            {"role": "system", "content": apet.meta_prompt_sys},
            {"role": "user", "content": meta},
        ]
        resp = get_llm_response(messages, model=self.model)
        return Prompt(resp.content.strip())

    def optimize(self, initial: Prompt, steps: int = 3) -> Tuple[Prompt, float]:
        best = initial
        best_score = self.evaluator.score(best)
        print(f"Initial score: {best_score:.4f}")
        for i in range(steps):
            candidate = self._propose(best)
            score = self.evaluator.score(candidate)
            print(f"Step {i + 1}: {score:.4f}")
            if score > best_score:
                best = candidate
                best_score = score
        return best, best_score


def main() -> None:
    evaluator = ClaudetteEvaluator()
    start_prompt = Prompt("Classify if the clause is fair or unfair. Respond with FAIR or UNFAIR.")
    optimizer = SimpleOPTS(evaluator)
    best, score = optimizer.optimize(start_prompt)
    print("Best instruction:", best.instruction)
    print("Validation accuracy:", score)


if __name__ == "__main__":
    main()
