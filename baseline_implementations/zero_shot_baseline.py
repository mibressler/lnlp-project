from typing import Optional

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from utils.dataset import ClaudetteDataset
from utils.llm import get_llm_response
from utils.metrics import compute_binary_metrics, compute_multilabel_metrics, display_metrics


def _split_labels(s: str):
    if pd.isnull(s):
        return []
    s = str(s).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def classify_text(text: str, label_type_options):
    binary_prompt = (
        "Given the following text, classify it as 0 (fair clause) or 1 (unfair clause):\n"
        "Only respond with a single digit: 0 or 1.\n\n"
        f"Text: {text}\n\nLabel (0 for fair, 1 for unfair):"
    )
    binary_resp = get_llm_response([
        {"role": "user", "content": binary_prompt}
    ])
    binary_label = binary_resp.content.strip()
    if binary_label not in ["0", "1"]:
        binary_label = "0"

    multilabel_prompt = (
        "Given the following text, which of these categories apply? "
        "Only respond with a comma-separated list of the category codes (a, ch, cr, j, law, ltd, ter, use). "
        "If none apply, respond with 'none'.\n\n"
        "Categories:\n"
        "Arbitration <a>\n"
        "- Unilateral change <ch>\n"
        "- Content removal <cr>\n"
        "- Jurisdiction <j>\n"
        "- Choice of law <law>\n"
        "- Limitation of liability <ltd>\n"
        "- Unilateral termination <ter>\n"
        "- Contract by using <use>\n\n"
        "Each code refers to a type of clause that might be unfair in a different way.\n"
        f"Text: {text}\n\nCategories:"
    )
    multi_resp = get_llm_response([
        {"role": "user", "content": multilabel_prompt}
    ])
    raw = multi_resp.content.strip().lower()
    if raw == "none":
        multilabels = []
    else:
        multilabels = [
            x.strip()
            for x in raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
            if x.strip() in label_type_options
        ]
    return int(binary_label), multilabels


def run(dataset: Optional[ClaudetteDataset] = None, *, sample_size: int = 200) -> None:
    """Run zero-shot LLM baseline using OpenRouter and print metrics."""

    dataset = dataset or ClaudetteDataset()
    df = dataset.get_dataset("test").sample(sample_size, random_state=42)
    train = dataset.get_dataset("train")

    mlb = MultiLabelBinarizer()
    mlb.fit(train["label_type"].apply(_split_labels))
    label_type_options = [x.lower() for x in mlb.classes_]

    binary_preds = []
    multilabel_preds = []
    for text in tqdm(df["text"], desc="Classifying with LLM"):
        binary, multilabel = classify_text(text, label_type_options)
        binary_preds.append(binary)
        multilabel_preds.append(multilabel)

    binary_metrics = compute_binary_metrics(df["label"], binary_preds)
    y_true_multi = mlb.transform(df["label_type"].apply(_split_labels))
    y_pred_multi = mlb.transform(multilabel_preds)
    multi_metrics = compute_multilabel_metrics(y_true_multi, y_pred_multi, label_type_options)

    display_metrics("LLM zero-shot - Test", binary_metrics, multi_metrics)
