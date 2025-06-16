from typing import Dict, Iterable, Optional

import pandas as pd

import numpy as np
from sklearn.metrics import classification_report


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return accuracy, micro and macro F1 for binary classification."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    support = len(list(y_true))
    return {
        "accuracy": report.get("accuracy", 0.0),
        "micro_f1": report.get("accuracy", 0.0),  # For binary tasks accuracy == micro F1
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "precision": report.get("weighted avg", {}).get("precision", 0.0),
        "recall": report.get("weighted avg", {}).get("recall", 0.0),
        "support": support,
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Return detailed metrics for multi-label classification."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    per_label = {}
    for label, vals in report.items():
        if label in {"micro avg", "macro avg", "weighted avg", "samples avg", "accuracy"}:
            continue
        per_label[label] = {
            "precision": vals.get("precision", 0.0),
            "recall": vals.get("recall", 0.0),
            "f1": vals.get("f1-score", 0.0),
            "support": vals.get("support", 0),
        }

    support = len(y_true)

    micro = report.get("micro avg", {})
    macro = report.get("macro avg", {})

    return {
        "micro_f1": micro.get("f1-score", 0.0),
        "macro_f1": macro.get("f1-score", 0.0),
        "micro_precision": micro.get("precision", 0.0),
        "micro_recall": micro.get("recall", 0.0),
        "macro_precision": macro.get("precision", 0.0),
        "macro_recall": macro.get("recall", 0.0),
        "per_label": per_label,
        "support": support,
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
            if k == "per_label":
                print("Per-label metrics:")
                for label, scores in v.items():
                    print(
                        f"  {label}: p={scores['precision']:.4f} r={scores['recall']:.4f} f1={scores['f1']:.4f} s={scores['support']}"
                    )
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
        ml_data = [(k, v) for k, v in multilabel.items() if k != "per_label"]
        if ml_data:
            df_m = pd.DataFrame(ml_data, columns=["Metric", "Value"])
            display(df_m.style.format({"Value": "{:.4f}"}).hide_index())
        per_label = multilabel.get("per_label")
        if per_label:
            df_l = pd.DataFrame(
                [
                    {
                        "Label": lbl,
                        "Precision": vals.get("precision", 0.0),
                        "Recall": vals.get("recall", 0.0),
                        "F1": vals.get("f1", 0.0),
                        "Support": vals.get("support", 0),
                    }
                    for lbl, vals in per_label.items()
                ]
            )
            display(
                df_l.style.format(
                    {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}
                ).hide_index()
            )
    return None


