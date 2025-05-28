import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# Helper to split and clean label_type strings into lists
def split_labels(s):
    if pd.isnull(s):
        return []
    s = str(s).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [x.strip().lower() for x in s.split(",") if x.strip()]

# Load datasets
train = pd.read_csv(r"r:\cl-2\michael\claudette_train_merged.tsv", sep="\t")
val = pd.read_csv(r"r:\cl-2\michael\claudette_val_merged.tsv", sep="\t")
test = pd.read_csv(r"r:\cl-2\michael\claudette_test_merged.tsv", sep="\t")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])
X_test = vectorizer.transform(test["text"])

# Binary classification: label
y_train_label = train["label"]
y_val_label = val["label"]
y_test_label = test["label"]

clf_label = LinearSVC(max_iter=1000, class_weight="balanced")
clf_label.fit(X_train, y_train_label)

print("Binary classification (label) with SVM:")
print("Validation:")
val_pred = clf_label.predict(X_val)
val_report = classification_report(y_val_label, val_pred, output_dict=True)
print(classification_report(y_val_label, val_pred))
print("Validation Micro F1:", val_report['accuracy'])  # For binary, accuracy == micro F1
print("Validation Macro F1:", val_report['macro avg']['f1-score'])

print("Test:")
test_pred = clf_label.predict(X_test)
test_report = classification_report(y_test_label, test_pred, output_dict=True)
print(classification_report(y_test_label, test_pred))
print("Test Micro F1:", test_report['accuracy'])  # For binary, accuracy == micro F1
print("Test Macro F1:", test_report['macro avg']['f1-score'])

# Multilabel classification: label_type
mlb = MultiLabelBinarizer()
y_train_type = mlb.fit_transform(train["label_type"].apply(split_labels))
y_val_type = mlb.transform(val["label_type"].apply(split_labels))
y_test_type = mlb.transform(test["label_type"].apply(split_labels))

clf_type = OneVsRestClassifier(LinearSVC(max_iter=1000, class_weight="balanced"))
clf_type.fit(X_train, y_train_type)

print("Multilabel classification (label_type) with SVM:")
print("Validation:")
val_pred_type = clf_type.predict(X_val)
val_report_type = classification_report(y_val_type, val_pred_type, target_names=mlb.classes_, zero_division=0, output_dict=True)
print(classification_report(y_val_type, val_pred_type, target_names=mlb.classes_, zero_division=0))
print("Validation Micro F1:", val_report_type['micro avg']['f1-score'])
print("Validation Macro F1:", val_report_type['macro avg']['f1-score'])

print("Test:")
test_pred_type = clf_type.predict(X_test)
test_report_type = classification_report(y_test_type, test_pred_type, target_names=mlb.classes_, zero_division=0, output_dict=True)
print(classification_report(y_test_type, test_pred_type, target_names=mlb.classes_, zero_division=0))
print("Test Micro F1:", test_report_type['micro avg']['f1-score'])
print("Test Macro F1:", test_report_type['macro avg']['f1-score'])