from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional
import pandas as pd

from utils.dataset import ClaudetteDataset
from utils.metrics import compute_binary_metrics, compute_multilabel_metrics, display_metrics


def _split_labels(s: str):
    if pd.isnull(s):
        return []
    s = str(s).replace('[', '').replace(']', '').replace("'", '').replace('"', '')
    return [x.strip().lower() for x in s.split(',') if x.strip()]


def run_logistic(
    dataset: Optional[ClaudetteDataset] = None,
    *,
    max_features: int = 10000,
    sample_size: Optional[int] = None,
):
    """Run TF-IDF + Logistic Regression baseline and return metrics."""
    dataset = dataset or ClaudetteDataset()
    train = dataset.get_dataset('train')
    val = dataset.get_dataset('val')
    test = dataset.get_dataset('test')
    if sample_size:
        test = test.sample(sample_size, random_state=42)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    X_test = vectorizer.transform(test['text'])

    y_train_b = train['label']
    y_val_b = val['label']
    y_test_b = test['label']
    clf_b = LogisticRegression(max_iter=1000)
    clf_b.fit(X_train, y_train_b)
    val_pred = clf_b.predict(X_val)
    test_pred = clf_b.predict(X_test)

    binary_val_metrics = compute_binary_metrics(y_val_b, val_pred)
    binary_test_metrics = compute_binary_metrics(y_test_b, test_pred)

    mlb = MultiLabelBinarizer()
    y_train_m = mlb.fit_transform(train['label_type'].apply(_split_labels))
    y_val_m = mlb.transform(val['label_type'].apply(_split_labels))
    y_test_m = mlb.transform(test['label_type'].apply(_split_labels))

    clf_m = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf_m.fit(X_train, y_train_m)
    val_pred_m = clf_m.predict(X_val)
    test_pred_m = clf_m.predict(X_test)

    multi_val_metrics = compute_multilabel_metrics(y_val_m, val_pred_m, mlb.classes_)
    multi_test_metrics = compute_multilabel_metrics(y_test_m, test_pred_m, mlb.classes_)

    display_metrics('TF-IDF Logistic - Validation', binary_val_metrics, multi_val_metrics)
    display_metrics('TF-IDF Logistic - Test', binary_test_metrics, multi_test_metrics)

    params = f"model=logistic max_features={max_features}"
    sample_n = len(test)
    return {
        "binary_val": binary_val_metrics,
        "binary_test": binary_test_metrics,
        "multi_val": multi_val_metrics,
        "multi_test": multi_test_metrics,
        "params": params,
        "sample_size": sample_n,
    }


def run_svm(
    dataset: Optional[ClaudetteDataset] = None,
    *,
    max_features: int = 10000,
    sample_size: Optional[int] = None,
):
    """Run TF-IDF + Linear SVM baseline and return metrics."""
    dataset = dataset or ClaudetteDataset()
    train = dataset.get_dataset('train')
    val = dataset.get_dataset('val')
    test = dataset.get_dataset('test')
    if sample_size:
        test = test.sample(sample_size, random_state=42)

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    X_test = vectorizer.transform(test['text'])

    y_train_b = train['label']
    y_val_b = val['label']
    y_test_b = test['label']
    clf_b = LinearSVC(max_iter=1000, class_weight='balanced')
    clf_b.fit(X_train, y_train_b)
    val_pred = clf_b.predict(X_val)
    test_pred = clf_b.predict(X_test)

    binary_val_metrics = compute_binary_metrics(y_val_b, val_pred)
    binary_test_metrics = compute_binary_metrics(y_test_b, test_pred)

    mlb = MultiLabelBinarizer()
    y_train_m = mlb.fit_transform(train['label_type'].apply(_split_labels))
    y_val_m = mlb.transform(val['label_type'].apply(_split_labels))
    y_test_m = mlb.transform(test['label_type'].apply(_split_labels))

    clf_m = OneVsRestClassifier(LinearSVC(max_iter=1000, class_weight='balanced'))
    clf_m.fit(X_train, y_train_m)
    val_pred_m = clf_m.predict(X_val)
    test_pred_m = clf_m.predict(X_test)

    multi_val_metrics = compute_multilabel_metrics(y_val_m, val_pred_m, mlb.classes_)
    multi_test_metrics = compute_multilabel_metrics(y_test_m, test_pred_m, mlb.classes_)

    display_metrics('TF-IDF SVM - Validation', binary_val_metrics, multi_val_metrics)
    display_metrics('TF-IDF SVM - Test', binary_test_metrics, multi_test_metrics)

    params = f"model=svm max_features={max_features}"
    sample_n = len(test)
    return {
        "binary_val": binary_val_metrics,
        "binary_test": binary_test_metrics,
        "multi_val": multi_val_metrics,
        "multi_test": multi_test_metrics,
        "params": params,
        "sample_size": sample_n,
    }
