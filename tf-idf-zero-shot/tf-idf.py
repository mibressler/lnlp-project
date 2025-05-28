import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# Helper to split label_type strings into lists
def split_labels(s):
    if pd.isnull(s):
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

# Load datasets
train = pd.read_csv(r"r:\cl-2\tf-idf\claudette_train_merged.tsv", sep="\t")
val = pd.read_csv(r"r:\cl-2\tf-idf\claudette_val_merged.tsv", sep="\t")
test = pd.read_csv(r"r:\cl-2\tf-idf\claudette_test_merged.tsv", sep="\t")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train["text"])
X_val = vectorizer.transform(val["text"])
X_test = vectorizer.transform(test["text"])

# Binary classification: label
y_train_label = train["label"]
y_val_label = val["label"]
y_test_label = test["label"]

clf_label = LogisticRegression(max_iter=1000)
clf_label.fit(X_train, y_train_label)

print("Binary classification (label):")
print("Validation:")
print(classification_report(y_val_label, clf_label.predict(X_val)))
print("Test:")
print(classification_report(y_test_label, clf_label.predict(X_test)))

# Multilabel classification: label_type
mlb = MultiLabelBinarizer()
y_train_type = mlb.fit_transform(train["label_type"].apply(split_labels))
y_val_type = mlb.transform(val["label_type"].apply(split_labels))
y_test_type = mlb.transform(test["label_type"].apply(split_labels))

clf_type = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf_type.fit(X_train, y_train_type)

print("Multilabel classification (label_type):")
print("Validation:")
print(classification_report(y_val_type, clf_type.predict(X_val), target_names=mlb.classes_))
print("Test:")
print(classification_report(y_test_type, clf_type.predict(X_test), target_names=mlb.classes_))

Fair is a label