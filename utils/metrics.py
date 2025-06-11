from typing import Dict, Iterable, Optional

import pandas as pd

import numpy as np
from sklearn.metrics import classification_report


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return accuracy, micro and macro F1 for binary classification."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report.get("accuracy", 0.0),
        "micro_f1": report.get("accuracy", 0.0),  # For binary tasks accuracy == micro F1
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "precision": report.get("weighted avg", {}).get("precision", 0.0),
        "recall": report.get("weighted avg", {}).get("recall", 0.0),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Return micro/macro F1 and per-label F1 for multi-label classification."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    per_label = {
        label: vals["f1-score"]
        for label, vals in report.items()
        if label not in {"micro avg", "macro avg", "weighted avg", "samples avg", "accuracy"}
    }

    return {
        "micro_f1": report.get("micro avg", {}).get("f1-score", 0.0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "per_label_f1": per_label,
    }


def display_metrics(title: str, binary: Optional[Dict[str, float]] = None, multilabel: Optional[Dict[str, object]] = None) -> None:
    """Pretty print classification metrics divided by task."""
    print(f"=== {title} ===")
    if binary is not None:
        print("-- Binary classification --")
        for k, v in binary.items():
            print(f"{k}: {v:.4f}")
    if multilabel is not None:
        print("-- Multilabel classification --")
        for k, v in multilabel.items():
            if k == "per_label_f1":
                print("Per-label F1:")
                for label, score in v.items():
                    print(f"  {label}: {score:.4f}")
            else:
                print(f"{k}: {v:.4f}")
    print()


def display_metrics_table(
    title: str,
    binary: Optional[Dict[str, float]] = None,
    multilabel: Optional[Dict[str, object]] = None,
) -> None:
    """Display metrics as nicely formatted tables in a Jupyter notebook."""
    try:
        from IPython.display import display, HTML
    except Exception:
        # Fallback to the plain text version if IPython is not available
        display_metrics(title, binary, multilabel)
        return

    display(HTML(f"<h3>{title}</h3>"))

    if binary is not None:
        display(HTML("<strong>Binary classification</strong>"))
        df_b = pd.DataFrame(list(binary.items()), columns=["Metric", "Value"])
        display(df_b.style.format({"Value": "{:.4f}"}).hide_index())

    if multilabel is not None:
        display(HTML("<strong>Multilabel classification</strong>"))
        ml_data = [(k, v) for k, v in multilabel.items() if k != "per_label_f1"]
        if ml_data:
            df_m = pd.DataFrame(ml_data, columns=["Metric", "Value"])
            display(df_m.style.format({"Value": "{:.4f}"}).hide_index())
        per_label = multilabel.get("per_label_f1")
        if per_label:
            df_l = pd.DataFrame(
                list(per_label.items()), columns=["Label", "F1"]
            )
            display(df_l.style.format({"F1": "{:.4f}"}).hide_index())

