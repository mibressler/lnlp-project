import os
import pandas as pd
import openai
from tqdm import tqdm

openai.api_key = "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"  # <-- Replace with your actual OpenAI API key

# Helper for LLM call
def classify_text(text, label_type_options):
    # Binary label
    binary_prompt = (
        "Given the following text, classify it as 0 (fair clause) or 1 (unfair clause):\n"
        "Only respond with a single digit: 0 or 1.\n\n"
        f"Text: {text}\n\nLabel (0 for fair, 1 for unfair):"
    )
    binary_resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": binary_prompt}],
        max_tokens=1,
        temperature=0
    )
    binary_label = binary_resp.choices[0].message.content.strip()
    if binary_label not in ["0", "1"]:
        binary_label = "0"  # fallback

    # Multilabel
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
    multi_resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": multilabel_prompt}],
        max_tokens=32,
        temperature=0
    )
    raw = multi_resp.choices[0].message.content.strip().lower()
    if raw == "none":
        multilabels = []
    else:
        multilabels = [x.strip() for x in raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",") if x.strip() in label_type_options]
    return binary_label, multilabels

# Load data (use a larger sample for cost reasons)
df = pd.read_csv(r"r:\cl-2\michael\claudette_test_merged.tsv", sep="\t").sample(200, random_state=42)

# Get all possible label_type categories from your train set
train = pd.read_csv(r"r:\cl-2\michael\claudette_train_merged.tsv", sep="\t")
from sklearn.preprocessing import MultiLabelBinarizer

def split_labels(s):
    if pd.isnull(s):
        return []
    s = str(s).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [x.strip().lower() for x in s.split(",") if x.strip()]

mlb = MultiLabelBinarizer()
mlb.fit(train["label_type"].apply(split_labels))
label_type_options = [x.lower() for x in mlb.classes_]

# Run zero-shot classification
binary_preds = []
multilabel_preds = []

for text in tqdm(df["text"], desc="Classifying with LLM"):
    binary, multilabel = classify_text(text, label_type_options)
    binary_preds.append(int(binary))
    multilabel_preds.append(multilabel)

# Evaluate
from sklearn.metrics import classification_report

print("Binary classification (LLM zero-shot):")
binary_report = classification_report(df["label"], binary_preds, output_dict=True)
print(classification_report(df["label"], binary_preds))
print("Binary Micro F1:", binary_report['accuracy'])  # For binary, accuracy == micro F1
print("Binary Macro F1:", binary_report['macro avg']['f1-score'])

print("Multilabel classification (LLM zero-shot):")
y_true_multi = mlb.transform(df["label_type"].apply(split_labels))
y_pred_multi = mlb.transform(multilabel_preds)
multi_report = classification_report(
    y_true_multi, y_pred_multi, target_names=label_type_options, zero_division=0, output_dict=True
)
print(classification_report(
    y_true_multi, y_pred_multi, target_names=label_type_options, zero_division=0
))
print("Multilabel Micro F1:", multi_report['micro avg']['f1-score'])
print("Multilabel Macro F1:", multi_report['macro avg']['f1-score'])


print("label_type_options:", label_type_options)
print("Sample LLM output:", multilabel_preds[:5])
print("Sample true labels:", df["label_type"].apply(split_labels).tolist()[:5])