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
from uuid import uuid4

# New imports for analysis and charts
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.formula.api import ols
import statsmodels.api as sm

csv.field_size_limit(10_000_000)

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ========== Constants ============
INSTRUCTION_STRATEGIES_WELL_PERFORMING = [
    "Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.",
    "Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding \"Let's think step-by-step\".",
    "Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.",
    "Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.",
    "At the end of the prompt, add a phrase that evokes a strong emotion. When doing so, keep the following four points in mind:\n1. Define emotional goals: Identify the emotional response you want to evoke, such as encouragement, motivation, or reassurance.\n2. Use positive language: Incorporate words and phrases that are positive and supportive. Examples include \"believe in your abilities,\" \"excellent,\" \"success,\" and \"outstanding achievements\".\n3. Emphasize key words: Use techniques like exclamation marks and capitalized words to highlight important aspects and to enhance the emotional impact.\n4. Incorporate social and self-esteem cues: Design stimuli that leverage social influence (e.g., group membership, others' opinions) and boost self-esteem and motivation. This can help regulate the emotional response of the Large Language Models and tap into intrinsic motivation.",
    "For a given prompt, add a phrase such as \"Read the question again\" that instructs the Large Language Models to reread the question before generating an answer. This strategy is particularly effective for complex tasks and helps enhance the quality and reliability of the model's outputs",
    "Clearly define the desired style in the given prompt. For example, you might say, \"Write a formal letter about...\" or \"Create a casual conversation discussing...\". This guidance helps the model produce text that matches the requested stylistic elements, whether it's formal, informal, technical, or poetic.",
    "For a given prompt, add a phrase that instructs the Large Language Models to rephrase the question before responding, such as \"Rephrase and expand the question, and respond.\"",
    "Make the description of the given prompt more specific. This makes it easier for Large Language Models to correctly execute prompt instructions.",
    "To allow Large Language Models to make logical and unbiased inferences, add phrases to a given prompt that instruct it to remove opinionated content. This helps the model concentrate on providing responses based on careful analysis and logical reasoning, minimizing biases.",
    "If a given prompt has long instructions, make it shorter by condensing it to only the essential parts. Never remove the instruction to strictly respond with '0' for fair or '1' for unfair.",
    "Edit the prompt instruction to invoke legal reasoning for problem solving: 1) State the goal of determining the unfairness of a clause. 2) Give a detailed definition of what could be considered an unfair clause. 3) compare the given sentence with the definition to estimate which parts of the sentence falls under that definition. 4) make a final determination based on the comparison."
    "Craft a concise description of the most capable expert for the task, addressing them in second person (e.g., 'You are an expert in...') to enhance precision and focus the model's response.",
    "Guide the model through step-by-step reasoning by adding a precise phrase like 'Let's think step-by-step' at the end, ensuring explanations are logical and concise while shortening unnecessary elaboration.",
    "Envision three experts collaboratively solving the problem: each briefly shares one step of thinking per round, and any who realize they're wrong exit immediately. This promotes precise, error-minimizing reasoning without verbose discussions.",
    "Ensure all essential information is embedded succinctly in the prompt, adding only what's needed to clarify without altering the objective, thereby making the instruction more precise and shorter.",
    "Append a brief phrase evoking positive emotion (e.g., 'Achieve outstanding success!') to motivate the model. Focus on: 1) Targeting encouragement or reassurance; 2) Using supportive words like 'excellent' or 'believe'; 3) Emphasizing with exclamation or capitals; 4) Boosting self-esteem via motivational cues. Keep it concise to avoid lengthening the prompt.",
    "Add a short directive like 'Read the question again carefully' before responding, improving accuracy for complex tasks by encouraging precise comprehension without unnecessary repetition.",
    "Specify the desired style succinctly in the prompt (e.g., 'Write in a formal tone...' or 'Use poetic language for...'), ensuring the instruction is precise and guides the model to match the style efficiently.",
    "Instruct the model to 'Rephrase the question concisely, then respond,' promoting clearer understanding and more focused, precise answers while avoiding verbose expansions.",
    "Refine the prompt's description to be more specific and concise, eliminating ambiguities to help the model execute instructions accurately and efficiently.",
    "Add a neutral directive like 'Base your response on logical reasoning only, avoiding opinions or biases' to foster unbiased, precise inferences focused on analysis.",
    "For lengthy instructions, condense to essential elements only, prioritizing clarity and brevity while preserving core objectives and never removing requirements like strictly responding with '0' for fair or '1' for unfair.",
    "Revise the prompt to invoke precise legal reasoning: 1) State the goal of assessing clause unfairness briefly; 2) Provide a concise definition of unfair clauses; 3) Compare the sentence directly to the definition, highlighting matching elements succinctly; 4) Conclude with a clear '0' (fair) or '1' (unfair) determination based on the comparison."
    "Completely rewrite the instruction from scratch"
]

INSTRUCTION_STRATEGIES_ORIGINAL = [
    "Craft a concise description of the most capable expert for the task, addressing them in second person (e.g., 'You are an expert in...') to enhance precision and focus the model's response.",
    "Guide the model through step-by-step reasoning by adding a precise phrase like 'Let's think step-by-step' at the end, ensuring explanations are logical and concise while shortening unnecessary elaboration.",
    "Envision three experts collaboratively solving the problem: each briefly shares one step of thinking per round, and any who realize they're wrong exit immediately. This promotes precise, error-minimizing reasoning without verbose discussions.",
    "Ensure all essential information is embedded succinctly in the prompt, adding only what's needed to clarify without altering the objective, thereby making the instruction more precise and shorter.",
    "Append a brief phrase evoking positive emotion (e.g., 'Achieve outstanding success!') to motivate the model. Focus on: 1) Targeting encouragement or reassurance; 2) Using supportive words like 'excellent' or 'believe'; 3) Emphasizing with exclamation or capitals; 4) Boosting self-esteem via motivational cues. Keep it concise to avoid lengthening the prompt.",
    "Add a short directive like 'Read the question again carefully' before responding, improving accuracy for complex tasks by encouraging precise comprehension without unnecessary repetition.",
    "Specify the desired style succinctly in the prompt (e.g., 'Write in a formal tone...' or 'Use poetic language for...'), ensuring the instruction is precise and guides the model to match the style efficiently.",
    "Instruct the model to 'Rephrase the question concisely, then respond,' promoting clearer understanding and more focused, precise answers while avoiding verbose expansions.",
    "Refine the prompt's description to be more specific and concise, eliminating ambiguities to help the model execute instructions accurately and efficiently.",
    "Add a neutral directive like 'Base your response on logical reasoning only, avoiding opinions or biases' to foster unbiased, precise inferences focused on analysis.",
    "For lengthy instructions, condense to essential elements only, prioritizing clarity and brevity while preserving core objectives and never removing requirements like strictly responding with '0' for fair or '1' for unfair.",
    "Revise the prompt to invoke precise legal reasoning: 1) State the goal of assessing clause unfairness briefly; 2) Provide a concise definition of unfair clauses; 3) Compare the sentence directly to the definition, highlighting matching elements succinctly; 4) Conclude with a clear '0' (fair) or '1' (unfair) determination based on the comparison.",
    "Improve the instruction",
]

INSTRUCTION_STRATEGIES_LEGAL = [
    "Craft a concise second-person description of a highly capable legal expert specializing in consumer contract fairness, emphasizing expertise in EU Directive 93/13 and unfair terms analysis to focus the model's response (e.g., 'You are a seasoned legal scholar in EU consumer law...').",
    "Incorporate step-by-step legal reasoning by adding 'Let's analyze step-by-step' or similar, ensuring the model explains its thought process logically, compares the clause to unfairness definitions, and concludes with '0' or '1' while keeping explanations concise.",
    "Envision a panel of three legal experts (e.g., a judge, a consumer rights advocate, and a contract law professor) collaboratively analyzing the clause: each shares one brief reasoning step per round, exiting if wrong, to minimize errors and promote precise classification.",
    "Embed all essential legal information succinctly, such as key definitions from Directive 93/13 and examples of unfair terms, adding clarifications only if needed without changing the objective of binary '0' (fair) or '1' (unfair) classification.",
    "Append a brief motivational phrase evoking confidence and precision (e.g., 'Deliver an EXCELLENT, accurate verdict!'), using positive language, emphasis, and self-esteem cues to encourage reliable outputs without extending the prompt length.",
    "Add a directive like 'Carefully reread the clause and context before classifying' to promote thorough comprehension and accuracy in this complex legal task, enhancing response reliability.",
    "Specify a formal, analytical legal style (e.g., 'Respond in a precise judicial tone, concluding strictly with '0' or '1''), guiding the model to produce structured, professional classifications matching legal standards.",
    "Instruct the model to 'Concisely rephrase the clause and task, then classify,' fostering deeper understanding, focused reasoning, and elimination of ambiguities for more accurate '0' or '1' outputs.",
    "Refine the instruction to be highly specific and concise, incorporating legal criteria like 'significant imbalance' and 'good faith,' while eliminating ambiguities and preserving the strict binary response requirement.",
    "Add a directive for unbiased, logic-based analysis (e.g., 'Base your classification solely on legal facts and Directive 93/13, avoiding any personal opinions'), minimizing biases and emphasizing careful clause-definition comparison.",
    "Condense lengthy instructions to essentials, integrating a structured legal reasoning framework: state the unfairness assessment goal, define unfair clauses briefly, compare the clause, and determine '0' or '1' based on matches.",
    "Completely rewrite the instruction from scratch, combining expert role, step-by-step analysis, and legal definitions into a streamlined prompt that ensures binary '0' or '1' responses with high accuracy.",
]

TEMPLATE_STRATEGIES = [
    "Enhance the description of relationships between template elements, for example explaining how the statutory context provides legal foundations, the contract context offers specific background, the instruction guides the process, and the clause is the target for classification.",
    "Reorder the template elements to optimize logical flow, for example presenting the statutory context first, followed by contract context, instruction, and clause or another arrangement that could be better.",
    "Incorporate separators, delimiters, or formatting emphasis (e.g., bold, italics) to improve readability and highlight key sections of the template.",
    "Experimentally completely omit one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially simplifying or enriching it while maintaining the classification task's integrity with <instruction> and <clause>.",
    "Experimentally re-add one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially enriching it while maintaining the classification task's integrity with <instruction> and <clause>.",
    "Improve the prompt template",
]

TEMPLATE_STRATEGIES_LEGAL_REASONING = [
    "Enhance the description of relationships between template elements, for example explaining how the statutory context provides legal foundations, the contract context offers specific background, the instruction guides the process, and the clause is the target for classification.",
    "Reorder the template elements to optimize logical flow, for example presenting the statutory context first, followed by contract context, instruction, and clause or another arrangement that could be better.",
    "Incorporate separators, delimiters, or formatting emphasis (e.g., bold, italics) to improve readability and highlight key sections of the template.",
    "Experimentally completely omit one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially simplifying or enriching it while maintaining the classification task's integrity with <instruction> and <clause>.",
    "Experimentally re-add one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially enriching it while maintaining the classification task's integrity with <instruction> and <clause>.",
    "Design the template to invoke legal reasoning for problem solving: Legal reasoining typicially involves 1) State the goal of determining the unfairness of a clause (what to assess). 2) Give a detailed definition of what could be considered an unfair clause. 3) compare the given sentence with the definition to estimate which parts of the sentence falls under that definition. 4) make a final determination based on the comparison."
]

PLACEHOLDER_STATUTORY_CONTEXT = (
    "According to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties' rights and obligations, to the detriment of the consumer. This general definition is further specified in the Annex to the Directive, containing an indicative and non-exhaustive list of the terms which may be regarded as unfair, as well as in a few dozen judgments of the Court of Justice of the EU (Micklitz and Reich 2014). Examples of unfair clauses encompass taking jurisdiction away from the consumer, limiting liability for damages on health and/or gross negligence, imposing obligatory arbitration in a country different from consumer's residence, etc. Loos and Luzak (2016) identified five categories of potentially unfair clauses often appearing in the terms of online services: 1) establishing jurisdiction for disputes in a country different than consumer's residence; 2) choice of a foreign law governing the contract; 3) limitation of liability; 4) the provider's right to unilaterally terminate the contract/access to the service; and 5) the provider's right to unilaterally modify the contract/the service. Our research has identified three additional categories: 6) requiring a consumer to undertake arbitration before the court proceedings can commence; 7) the provider retaining the right to unilaterally remove consumer content from the service, including in-app purchases; 8) having a consumer accept the agreement simply by using the service, not only without reading it, but even without having to click on 'I agree/I accept.'"
)

META_PROMPT_INSTR = """
Imagine yourself as an expert in prompting techniques for LLMs. Your expertise is broad and deep. Your job is to reformulate instructions with precision, optimizing for accurate responses in a legal classification task. The reformulated instruction MUST ensure responses are strictly '0' for fair or '1' for unfair.

Available technique: {strategy}

Reformulate the below instruction using the technique. Include ALL original information. ONLY return the reformulated instruction.

Original Instruction: {parent_instr}
"""

META_PROMPT_INSTR_2 = """
You are an expert prompt engineer gently applying the following transformation strategy to improve an instruction for a classification task. It is important that responses at all times only consist '0' for fair or '1' for unfair.\n
Strategy: {strategy} \n
Original Instruction: {parent_instr} \n
New Instruction:
"""

META_PROMPT_INSTR_3 = """
You are an expert prompt engineer gently applying the following transformation strategy to improve an instruction for a legal classification task (predicting the fairness of an individual clause from a ToS contract). It is important that responses at all times only consist of '0' for fair or '1' for unfair.\n
Strategy: {strategy} \n
Original Instruction: {parent_instr} \n
New Instruction:
"""

META_PROMPT_INSTR_4 = """
You are an expert prompt engineer gently applying the following transformation strategy to improve an instruction for a legal classification task (predicting the fairness of an individual clause from a ToS contract). It is important that responses at all times only consist of '0' for fair or '1' for unfair.

STRATEGY: 
{strategy}

ORIGINAL INSTRUCTION: 
{parent_instr}

NEW INSTRUCTION:
"""

META_PROMPT_TEMPLATE = """
You are an expert prompt engineer gently applying the following transformation strategy to improve a prompt template for a classification task. Ensure the template includes placeholders: At least <instruction> for the classification instruction and <clause> for the clause text. <contract_context> and <statutory_context> may or may not be part of the template. It is important that responses at all times only consist '0' for fair or '1' for unfair.

Strategy: {strategy}

Original Template: {parent_template}

New Template:
"""

META_PROMPT_TEMPLATE_2 = """
You are an expert prompt engineer gently applying the following transformation strategy to improve a prompt template for a classification task. Ensure the template includes placeholders: At least <instruction> for the classification instruction and <clause> for the clause text. <contract_context> and <statutory_context> may or may not be part of the template. It is important that the template does not interfere with the model responding only with '0' for fair and '1' for unfair for the classification task the template is used for. Please ONLY RETURN THE NEW TEMPLATE.\n
Strategy: {strategy} \n
Original Template: {parent_template} \n
New Template:
"""

META_PROMPT_TEMPLATE_3 = """
You are an expert prompt engineer gently applying the following transformation strategy to improve a prompt template for a legal classification task (predicting the fairness of an individual clause from a ToS contract). Ensure the template includes the placeholders: At least <instruction> for the classification instruction and <clause> for the clause text. <contract_context> and <statutory_context> may or may not be part of the template. All placeholders in brackets automatically get replaced by the actual data. It is important that the template does not interfere with the model responding only with '0' for fair and '1' for unfair for the classification task the template is used for. Please ONLY RETURN THE NEW TEMPLATE.

STRATEGY:
{strategy} 

ORIGIGNAL TEMPLATE:
{parent_template}

NEW TEMPLATE:
"""

BASE_INSTR_CORE = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1)"
BASE_INSTR_USING_CONTEXT = " using the {contexts} for better understanding"
BASE_INSTR_SUFFIX = ". Respond only with '0' or '1'."

BASE_TEMPLATE_CORE = "Instruction: <instruction>\nClause: <clause>"
BASE_TEMPLATE_STATUTORY = "\nStatutory Context: <statutory_context>"
BASE_TEMPLATE_CONTRACT = "\nContract Context: <contract_context>"

# ========== Mutation Logger ============
class MutationLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_id", "gen", "child_idx",
                    "instr_arm", "instr_strategy",
                    "template_arm", "template_strategy",
                    "selected_by_bandit_instr", "selected_by_bandit_template",
                    "parent_idx", "parent_score", "parent_max_topk",
                    "child_score", "reward",
                    "valid_predictions", "total_predictions", "adjusted_f1_macro"
                ])

    def log(self, row):
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

# ========== Classes ============
class BanditSelector:
    def __init__(self, strategies, name=""):
        self.name = name  # For logging distinction
        self.strategies = strategies + ["INACTION"]
        self.num_arms = len(self.strategies)
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
    
    def select_arm(self):
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        return int(np.argmax(samples))
    
    def update(self, arm, reward):
        logging.info(f"Updating {self.name} bandit arm {arm} ({self.strategies[arm]}): reward={reward}")
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

    def get_posterior(self):
        return list(zip(self.alphas.tolist(), self.betas.tolist()))

class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
        self.score = np.nan
        self.instr_arm = None
        self.template_arm = None  # Track both arms
    
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
    patterns = [r'(?<!\d)[01](?!\d)', r'\boxed{([01])}', r'Answer: ([01])']
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1]
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
        print(f"---- Sent in Batch {i // batch_size + 1} ----")
        print(formatted[0])
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
            'detailed_report_string': "No valid predictions to evaluate.",
            'adjusted_f1_macro': 0.0
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
        valid_ratio = len(y_pred) / len(y_pred_all)
        adjusted_score = f1_macro * valid_ratio
        print(f"⭐ Adjusted F1 Macro Score: {adjusted_score:.4f}")

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
            'detailed_report_string': detailed_report_string,
            'adjusted_f1_macro': adjusted_score
        }

        return adjusted_score, metrics
    
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
            'detailed_report_string': f"Skipping metrics due to error: {e}",
            'adjusted_f1_macro': 0.0
        }
        return 0.0, metrics

def mutate_instruction(parent_instr, strategy, llm):
    if strategy == "INACTION":
        return parent_instr
    prompt = META_PROMPT_INSTR_4.format(strategy=strategy, parent_instr=parent_instr)
    print(f"Mutating instruction with strategy: {strategy}")
    print(f"Prompt: {prompt}")
    return llm.query([prompt])[0]

def mutate_template(parent_template, template_strategy, llm, statutory_enabled, contract_enabled):
    if template_strategy == "INACTION":
        return parent_template
    prompt = META_PROMPT_TEMPLATE_3.format(strategy=template_strategy, parent_template=parent_template)
    if not statutory_enabled:
        prompt += "\nDo not include the <statutory_context> placeholder in the new template."
    if not contract_enabled:
        prompt += "\nDo not include the <contract_context> placeholder in the new template."
    print(f"Mutating template with strategy: {template_strategy}")
    print(f"Prompt: {prompt}")
    return llm.query([prompt])[0]

def mutate_prompt_ga(parent, instr_selector, template_selector, llm, use_bandit_instr, use_bandit_template, statutory_enabled, contract_enabled):
    if use_bandit_instr:
        instr_arm = instr_selector.select_arm()
    else:
        instr_arm = random.randint(0, instr_selector.num_arms - 1)
    instr_strategy = instr_selector.strategies[instr_arm]
    new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
    
    if use_bandit_template:
        template_arm = template_selector.select_arm()
    else:
        template_arm = random.randint(0, template_selector.num_arms - 1)
    template_strategy = template_selector.strategies[template_arm]
    new_template = mutate_template(parent.template, template_strategy, llm, statutory_enabled, contract_enabled)
    
    child = Prompt(new_instr, new_template)
    child.instr_arm = instr_arm
    child.template_arm = template_arm
    return child

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

def optimize_prompt(train_x, train_context, train_y, llm, generations=50, pop_size=10, train_sample_size=50, use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True):
    run_id = str(uuid4())[:8]
    os.makedirs("runs", exist_ok=True)
    mutation_log_path = os.path.join("runs", f"{run_id}_mutations.csv")
    logger = MutationLogger(mutation_log_path)

    base_instr = get_base_instr(statutory_context_enabled, contract_context_enabled)
    base_template = get_base_template(statutory_context_enabled, contract_context_enabled)
    
    population = [Prompt(base_instr, base_template) for _ in range(pop_size)]
    instr_selector = BanditSelector(INSTRUCTION_STRATEGIES_ORIGINAL, name="Instruction")
    template_selector = BanditSelector(TEMPLATE_STRATEGIES, name="Template")
    
    best_score = 0.0
    no_improve_gens = 0
    max_no_improve = 500
    
    # Track population stats per generation for plots
    gen_stats = []

    for gen in range(generations):
        logging.info(f"Starting generation {gen + 1}")
        print(f"============ Generation {gen + 1} ============")
        
        scores_and_metrics = [evaluate(p, train_x, train_context, train_y, llm, sample_size=train_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled) for p in tqdm(population, desc="Evaluating population")]
        scores = [s[0] for s in scores_and_metrics]
        for p, score in zip(population, scores):
            p.score = score if not np.isnan(score) else 0.0
        population.sort(key=lambda p: p.score, reverse=True)
        print("⭐⭐ Scores:", [p.score for p in population])

        gen_stats.append({
            "gen": gen,
            "best": population[0].score,
            "mean": float(np.mean([p.score for p in population])),
            "median": float(np.median([p.score for p in population])),
        })
        
        current_best = population[0].score
        if current_best > best_score:
            best_score = current_best
            no_improve_gens = 0
        else:
            no_improve_gens += 1
            if no_improve_gens >= max_no_improve:
                logging.info(f"Early stopping at generation {gen + 1} due to no improvement.")
                break
        
        top_k = population[:pop_size // 2]
        children = []
        parent_max = max(p.score for p in top_k)
        for child_idx in range(pop_size - len(top_k)):
            parent_idx = random.randrange(len(top_k))
            parent = top_k[parent_idx]

            # Select arms
            if use_bandit_instr:
                instr_arm = instr_selector.select_arm()
            else:
                instr_arm = random.randint(0, instr_selector.num_arms - 1)
            instr_strategy = instr_selector.strategies[instr_arm]

            if use_bandit_template:
                template_arm = template_selector.select_arm()
            else:
                template_arm = random.randint(0, template_selector.num_arms - 1)
            template_strategy = template_selector.strategies[template_arm]

            # Mutate
            new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
            new_template = mutate_template(parent.template, template_strategy, llm,
                                           statutory_context_enabled, contract_context_enabled)
            child = Prompt(new_instr, new_template)
            child.instr_arm = instr_arm
            child.template_arm = template_arm

            # Evaluate child
            child_score, child_metrics = evaluate(child, train_x, train_context, train_y, llm,
                                                  sample_size=train_sample_size,
                                                  statutory_enabled=statutory_context_enabled,
                                                  contract_enabled=contract_context_enabled)
            child.score = child_score

            # Reward and update bandits
            reward = 1 if child.score > parent_max else 0
            instr_selector.update(instr_arm, reward)
            template_selector.update(template_arm, reward)

            # Log mutation
            logger.log([
                run_id, gen, child_idx,
                instr_arm, instr_strategy,
                template_arm, template_strategy,
                int(use_bandit_instr), int(use_bandit_template),
                parent_idx, parent.score, parent_max,
                child.score, reward,
                child_metrics.get('valid_predictions', 0),
                child_metrics.get('total_predictions', 0),
                child_metrics.get('adjusted_f1_macro', 0.0)
            ])

            children.append(child)
        
        population = top_k + children
    
    best_prompt = max(population, key=lambda p: p.score)
    with open("best_prompt.json", "w", encoding="utf-8") as f:
        json.dump({"instr": best_prompt.instr, "template": best_prompt.template}, f)
    logging.info("Best prompt saved to best_prompt.json")
    
    logging.info(f"Best Instruction: {best_prompt.instr}")
    logging.info(f"Best Template: {best_prompt.template}")
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)

    # Save bandit posterior
    bandits_path = os.path.join("runs", f"{run_id}_bandits.json")
    with open(bandits_path, "w", encoding="utf-8") as f:
        json.dump({
            "instruction": {
                "strategies": instr_selector.strategies,
                "alphas": instr_selector.alphas.tolist(),
                "betas": instr_selector.betas.tolist()
            },
            "template": {
                "strategies": template_selector.strategies,
                "alphas": template_selector.alphas.tolist(),
                "betas": template_selector.betas.tolist()
            },
            "gen_stats": gen_stats
        }, f, indent=2)

    return best_prompt, run_id, mutation_log_path, bandits_path, gen_stats

# ========== Analysis helpers ============
def analyze_run(run_id, mutation_log_path, bandits_path, gen_stats, show_plots=True):
    print(f"\n=== Analysis for run_id={run_id} ===")
    if not os.path.exists(mutation_log_path):
        print("No mutation log found; skipping analysis.")
        return

    df = pd.read_csv(mutation_log_path)
    if df.empty:
        print("Mutation log is empty.")
        return

    # Derived columns
    df['delta'] = df['child_score'] - df['parent_score']

    # Summary tables
    print("\nInstruction Strategy Selection Frequency:")
    instr_freq = df.groupby('instr_strategy').size().sort_values(ascending=False)
    print(instr_freq)

    print("\nTemplate Strategy Selection Frequency:")
    templ_freq = df.groupby('template_strategy').size().sort_values(ascending=False)
    print(templ_freq)

    print("\nInstruction Success Rate (reward mean):")
    instr_success = df.groupby('instr_strategy')['reward'].mean().sort_values(ascending=False)
    print(instr_success)

    print("\nTemplate Success Rate (reward mean):")
    templ_success = df.groupby('template_strategy')['reward'].mean().sort_values(ascending=False)
    print(templ_success)

    print("\nInstruction Impact (delta score): count, mean, median")
    instr_impact = df.groupby('instr_strategy')['delta'].agg(['count','mean','median']).sort_values('mean', ascending=False)
    print(instr_impact)

    print("\nTemplate Impact (delta score): count, mean, median")
    templ_impact = df.groupby('template_strategy')['delta'].agg(['count','mean','median']).sort_values('mean', ascending=False)
    print(templ_impact)

    # Save tables to CSV
    out_base = os.path.join("runs", f"{run_id}")
    instr_freq.to_csv(out_base + "_instr_freq.csv")
    templ_freq.to_csv(out_base + "_templ_freq.csv")
    instr_success.to_csv(out_base + "_instr_success.csv")
    templ_success.to_csv(out_base + "_templ_success.csv")
    instr_impact.to_csv(out_base + "_instr_impact.csv")
    templ_impact.to_csv(out_base + "_templ_impact.csv")

    # Regression: effect controlling for parent_score
    if df['instr_strategy'].nunique() > 1:
        try:
            model = ols('delta ~ C(instr_strategy) + parent_score', data=df).fit(cov_type="HC3")
            print("\nRegression (instruction): delta ~ C(instr_strategy) + parent_score")
            print(model.summary())
            with open(out_base + "_instr_regression.txt", "w", encoding="utf-8") as f:
                f.write(model.summary().as_text())
        except Exception as e:
            print(f"Instruction regression failed: {e}")

    if df['template_strategy'].nunique() > 1:
        try:
            model_t = ols('delta ~ C(template_strategy) + parent_score', data=df).fit(cov_type="HC3")
            print("\nRegression (template): delta ~ C(template_strategy) + parent_score")
            print(model_t.summary())
            with open(out_base + "_templ_regression.txt", "w", encoding="utf-8") as f:
                f.write(model_t.summary().as_text())
        except Exception as e:
            print(f"Template regression failed: {e}")

    # Sequences and transitions
    df_sorted = df.sort_values(['gen', 'child_idx'])
    instr_seq = df_sorted['instr_strategy'].tolist()
    templ_seq = df_sorted['template_strategy'].tolist()

    instr_bigrams = Counter(zip(instr_seq[:-1], instr_seq[1:]))
    templ_bigrams = Counter(zip(templ_seq[:-1], templ_seq[1:]))

    print("\nTop 10 instruction bigrams (strategy transitions):")
    print(instr_bigrams.most_common(10))

    print("\nTop 10 template bigrams (strategy transitions):")
    print(templ_bigrams.most_common(10))

    # Transition matrices (heatmaps)
    def plot_transition_heatmap(seq, title, filename):
        uniq = list(pd.unique(seq))
        idx = {s: i for i, s in enumerate(uniq)}
        mat = np.zeros((len(uniq), len(uniq)), dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            mat[idx[a], idx[b]] += 1.0
        # Normalize rows
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums
        plt.figure(figsize=(max(8, len(uniq)*0.5), max(6, len(uniq)*0.5)))
        sns.heatmap(mat, xticklabels=uniq, yticklabels=uniq, cmap="Blues", annot=False)
        plt.title(title)
        plt.xlabel("Next strategy")
        plt.ylabel("Current strategy")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        if show_plots:
            plt.show()
        plt.close()

    plot_transition_heatmap(instr_seq, f"Instruction Strategy Transition Probabilities ({run_id})", out_base + "_instr_transitions.png")
    plot_transition_heatmap(templ_seq, f"Template Strategy Transition Probabilities ({run_id})", out_base + "_templ_transitions.png")

    # Score trajectories over generations
    if gen_stats:
        gs = pd.DataFrame(gen_stats)
        plt.figure(figsize=(8,5))
        plt.plot(gs['gen'], gs['best'], label='Best score')
        plt.plot(gs['gen'], gs['mean'], label='Mean score')
        plt.plot(gs['gen'], gs['median'], label='Median score')
        plt.xlabel("Generation")
        plt.ylabel("Adjusted F1 Macro")
        plt.title(f"Population Scores per Generation ({run_id})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base + "_gen_scores.png", dpi=150)
        if show_plots:
            plt.show()
        plt.close()

    # Bandit posterior visualization
    try:
        with open(bandits_path, "r", encoding="utf-8") as f:
            bd = json.load(f)

        for which in ["instruction", "template"]:
            strategies = bd[which]["strategies"]
            alphas = np.array(bd[which]["alphas"])
            betas = np.array(bd[which]["betas"])
            means = alphas / (alphas + betas)
            # 95% Beta intervals
            lower = stats.beta.ppf(0.025, alphas, betas)
            upper = stats.beta.ppf(0.975, alphas, betas)

            dfb = pd.DataFrame({
                "strategy": strategies,
                "mean": means,
                "lower": lower,
                "upper": upper
            }).sort_values("mean", ascending=False)

            print(f"\nBandit posterior ({which}) - mean and 95% CI:")
            print(dfb)

            plt.figure(figsize=(10, max(5, len(strategies)*0.4)))
            sns.pointplot(data=dfb, y="strategy", x="mean", join=False, color="b", errorbar=None)
            # Add intervals manually
            for i, row in dfb.iterrows():
                plt.plot([row['lower'], row['upper']], [dfb.index.get_loc(i)]*2, color='b')
            plt.title(f"Bandit Posterior Means with 95% CI ({which}, {run_id})")
            plt.xlabel("Posterior mean (alpha/(alpha+beta))")
            plt.ylabel("Strategy")
            plt.tight_layout()
            plt.savefig(out_base + f"_{which}_bandit_posterior.png", dpi=150)
            if show_plots:
                plt.show()
            plt.close()
    except Exception as e:
        print(f"Failed to plot bandit posterior: {e}")

    # Distribution plots for impact and success
    plt.figure(figsize=(10,6))
    order = instr_impact.index.tolist()
    sns.boxplot(data=df, x='instr_strategy', y='delta', order=order)
    plt.title(f"Instruction Strategy Impact (delta) ({run_id})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_base + "_instr_delta_box.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    order_t = templ_impact.index.tolist()
    sns.boxplot(data=df, x='template_strategy', y='delta', order=order_t)
    plt.title(f"Template Strategy Impact (delta) ({run_id})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_base + "_templ_delta_box.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # Bar charts for frequency and success rate
    plt.figure(figsize=(10,6))
    instr_freq.plot(kind='bar')
    plt.title(f"Instruction Strategy Selection Frequency ({run_id})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_base + "_instr_freq_bar.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    instr_success.plot(kind='bar')
    plt.title(f"Instruction Strategy Success Rate ({run_id})")
    plt.ylabel("Reward mean")
    plt.tight_layout()
    plt.savefig(out_base + "_instr_success_bar.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    templ_freq.plot(kind='bar')
    plt.title(f"Template Strategy Selection Frequency ({run_id})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_base + "_templ_freq_bar.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    templ_success.plot(kind='bar')
    plt.title(f"Template Strategy Success Rate ({run_id})")
    plt.ylabel("Reward mean")
    plt.tight_layout()
    plt.savefig(out_base + "_templ_success_bar.png", dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    print(f"\nAnalysis artifacts saved under runs/{run_id}_*.png and CSV tables.")

# ========== Main ============
def main(generations=20, pop_size=8, train_sample_size=10, test_sample_size=100, model_name="google/gemini-2.5-flash-lite-preview-06-17", use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True, show_plots=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = Data.load(os.path.join(base_dir, "train_unskewed.tsv"))
    test_data = Data.load(os.path.join(base_dir, "test.tsv"))
    
    llm = OpenRouterLLM(model_name)
    best_prompt, run_id, mutation_log_path, bandits_path, gen_stats = optimize_prompt(
        train_data.get_x(), train_data.get_context(), train_data.get_y(), llm,
        generations, pop_size, train_sample_size,
        use_bandit_instr, use_bandit_template,
        statutory_context_enabled, contract_context_enabled
    )
    
    print("\nRunning on test set...")
    test_f1, test_metrics = evaluate(best_prompt, test_data.get_x(), test_data.get_context(), test_data.get_y(), llm, sample_size=test_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled)
    print(f"Test Adjusted Macro F1: {test_f1:.4f}")
    logging.info(f"Test Adjusted Macro F1: {test_f1:.4f}")

    # Save final results
    result = {
        'run_id': run_id,
        'best_instruction': best_prompt.instr,
        'best_template': best_prompt.template,
        'test_metrics': test_metrics
    }
    with open(os.path.join("runs", f"{run_id}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Run analysis and produce charts/tables
    analyze_run(run_id, mutation_log_path, bandits_path, gen_stats, show_plots=show_plots)

    return result

if __name__ == "__main__":
    # You may adjust defaults here for faster runs during development
    main(
        generations=20,
        pop_size=8,
        train_sample_size=10,
        test_sample_size=100,
        model_name="google/gemini-2.5-flash-lite-preview-06-17",
        use_bandit_instr=True,
        use_bandit_template=True,
        statutory_context_enabled=True,
        contract_context_enabled=True,
        show_plots=True
    )