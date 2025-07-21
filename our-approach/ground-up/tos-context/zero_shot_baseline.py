import os
import random
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import openai
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import csv
import scipy.stats as stats
import logging  
import json  

csv.field_size_limit(10_000_000)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ========== Constants ============
PLACEHOLDER_STATUTORY_CONTEXT = (
    "According to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties' rights and obligations, to the detriment of the consumer. This general definition is further specified in the Annex to the Directive, containing an indicative and non-exhaustive list of the terms which may be regarded as unfair, as well as in a few dozen judgments of the Court of Justice of the EU (Micklitz and Reich 2014). Examples of unfair clauses encompass taking jurisdiction away from the consumer, limiting liability for damages on health and/or gross negligence, imposing obligatory arbitration in a country different from consumer's residence, etc. Loos and Luzak (2016) identified five categories of potentially unfair clauses often appearing in the terms of online services: 1) establishing jurisdiction for disputes in a country different than consumer's residence; 2) choice of a foreign law governing the contract; 3) limitation of liability; 4) the provider's right to unilaterally terminate the contract/access to the service; and 5) the provider's right to unilaterally modify the contract/the service. Our research has identified three additional categories: 6) requiring a consumer to undertake arbitration before the court proceedings can commence; 7) the provider retaining the right to unilaterally remove consumer content from the service, including in-app purchases; 8) having a consumer accept the agreement simply by using the service, not only without reading it, but even without having to click on 'I agree/I accept.'"
)

BASE_INSTR_CORE = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1)"
BASE_INSTR_USING_CONTEXT = " using the {contexts} for better understanding"
BASE_INSTR_SUFFIX = ". Respond only with '0' or '1'."

BASE_TEMPLATE_CORE = "Instruction: <instruction>\nClause: <clause>"
BASE_TEMPLATE_STATUTORY = "\nStatutory Context: <statutory_context>"
BASE_TEMPLATE_CONTRACT = "\nContract Context: <contract_context>"

# ========== Classes ============
class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
    
    def join_input(self, text, context, statutory_enabled, contract_enabled):
        statutory = PLACEHOLDER_STATUTORY_CONTEXT if statutory_enabled else ''
        contract = context if contract_enabled else ''
        return self.template.replace("<instruction>", self.instr).replace("<clause>", text).replace("<contract_context>", contract).replace("<statutory_context>", statutory)

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
    
    @staticmethod
    def load(path, delimiter="\t"):
        with open(path, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_ALL)
            lines = [row for row in reader if row]
        return Data(lines)
    
    def get_x(self):
        return [line[4] for line in self.dataset]
    
    def get_context(self):
        return [line[6] if len(line) > 6 else '' for line in self.dataset]
    
    def get_y(self):
        return [line[2] for line in self.dataset]

class OpenRouterLLM:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def query(self, prompts, temperature=0.7, max_tokens=256):
        outputs = []
        for prompt in prompts:
            retries = 0
            while retries < 3:
                try:
                    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    output = response.choices[0].message.content.strip()
                    outputs.append(output)
                    
                    # Nicely formatted print
                    print("\n=== SENT ===")
                    print(prompt)
                    print("\n=== RECEIVED ===")
                    print(output)
                    print("\n=============\n")
                    
                    break
                except openai.OpenAIError as e:  # More specific exception handling
                    logging.error(f"OpenAI API error: {e}")
                    time.sleep(2 ** retries)
                    retries += 1
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    time.sleep(2 ** retries)
                    retries += 1
            else:
                logging.warning("Max retries exceeded for prompt.")
                outputs.append("")  # Fallback empty response
        return outputs

# ========== Functions ============
def extract_answer(output):
    output = output.strip()
    if output in ['0', '1']:
        return output
    # Improved regex for better robustness (e.g., handles boxed answers)
    patterns = [r'(?<!\d)[01](?!\d)', r'\boxed{([01])}', r'Answer: ([01])']
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]  # Take last match
    logging.warning(f"Invalid answer: '{output}'")
    return 'invalid'

def evaluate(prompt_obj, data_x, data_context, data_y, llm, batch_size=20, sample_size=50, statutory_enabled=True, contract_enabled=True):
    n = len(data_x)
    if sample_size >= n:
        logging.info(f"Sample size {sample_size} >= dataset {n}. Using full dataset.")
        indices = list(range(n))
    else:
        indices = random.sample(range(n), sample_size)
    
    eval_x = [data_x[i] for i in indices]
    eval_context = [data_context[i] for i in indices]
    eval_y = [data_y[i] for i in indices]

    outputs = []
    for i in range(0, len(eval_x), batch_size):
        logging.info(f"Processing batch {i // batch_size + 1}")
        batch_x = eval_x[i:i+batch_size]
        batch_context = eval_context[i:i+batch_size]
        formatted = [prompt_obj.join_input(x, c, statutory_enabled, contract_enabled) for x, c in zip(batch_x, batch_context)]
        outputs.extend(llm.query(formatted, temperature=0.0))

    cleaned_outputs = [extract_answer(o) for o in outputs]
    y_true_all = [str(y).strip() for y in eval_y]
    y_pred_all = [p if p in ['0', '1'] else 'invalid' for p in cleaned_outputs]

    valid_indices = [i for i, (pred, true) in enumerate(zip(y_pred_all, y_true_all)) if pred in ['0', '1'] and true in ['0', '1']]
    y_true = [y_true_all[i] for i in valid_indices]
    y_pred = [y_pred_all[i] for i in valid_indices]

    logging.info(f"Valid predictions: {len(y_pred)} / {len(y_pred_all)}")

    if not y_pred:
        logging.warning("No valid predictions. Returning score of 0.")
        metrics = {
            'sample_size': 0,
            'valid_predictions': 0,
            'total_predictions': len(y_pred_all),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'support': {'0': 0, '1': 0},
            'classification_report': {},
            'unique_y_true': set(),
            'unique_y_pred': set(),
            'detailed_report_string': "No valid predictions to evaluate."
        }
        return 0.0, metrics

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
        support = {k: v['support'] for k, v in report.items() if k in ['0', '1']}
        detailed_report_string = classification_report(y_true, y_pred, digits=4, zero_division=0)

        metrics = {
            'sample_size': len(y_true),
            'valid_predictions': len(y_pred),
            'total_predictions': len(y_pred_all),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'support': support,
            'classification_report': report,
            'unique_y_true': set(y_true),
            'unique_y_pred': set(y_pred),
            'detailed_report_string': detailed_report_string
        }
        return f1_macro, metrics
    except ValueError as e:
        logging.error(f"Metrics error: {e}")
        metrics = {
            'sample_size': len(y_true),
            'valid_predictions': len(y_pred),
            'total_predictions': len(y_pred_all),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_micro': 0.0,
            'f1_macro': 0.0,
            'support': {'0': 0, '1': 0},
            'classification_report': {},
            'unique_y_true': set(y_true) if y_true else set(),
            'unique_y_pred': set(y_pred) if y_pred else set(),
            'detailed_report_string': f"Skipping metrics due to error: {e}"
        }
        return 0.0, metrics

def get_base_instr(statutory_enabled, contract_enabled):
    base_instr = BASE_INSTR_CORE
    contexts = []
    if statutory_enabled:
        contexts.append("statutory context")
    if contract_enabled:
        contexts.append("contract context")
    if contexts:
        base_instr += BASE_INSTR_USING_CONTEXT.format(contexts=' and '.join(contexts))
    base_instr += BASE_INSTR_SUFFIX
    return base_instr

def get_base_template(statutory_enabled, contract_enabled):
    base_template = BASE_TEMPLATE_CORE
    if statutory_enabled:
        base_template += BASE_TEMPLATE_STATUTORY
    if contract_enabled:
        base_template += BASE_TEMPLATE_CONTRACT
    return base_template

# ========== Main ============
def main(test_sample_size=100, model_name="google/gemini-2.5-flash-lite-preview-06-17", statutory_context_enabled=True, contract_context_enabled=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_data = Data.load(os.path.join(base_dir, "test.tsv"))
    
    llm = OpenRouterLLM(model_name)
    
    base_instr = get_base_instr(statutory_context_enabled, contract_context_enabled)
    base_template = get_base_template(statutory_context_enabled, contract_context_enabled)
    base_prompt = Prompt(base_instr, base_template)
    
    print("\nRunning zero-shot baseline on test set...")
    test_f1, test_metrics = evaluate(base_prompt, test_data.get_x(), test_data.get_context(), test_data.get_y(), llm, sample_size=test_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled)
    print(f"Test Macro F1: {test_f1:.4f}")
    logging.info(f"Test Macro F1: {test_f1:.4f}")

    result = {
        'best_instruction': base_instr,
        'best_template': base_template,
        'test_metrics': test_metrics
    }
    return result

if __name__ == "__main__":
    main()