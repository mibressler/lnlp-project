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
import pandas as pd  # New: For tables
import matplotlib.pyplot as plt  # New: For charts

csv.field_size_limit(10_000_000)

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Set random seed for reproducibility
#random.seed(42)
#np.random.seed(42)

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

# ========== Classes ============
class BanditSelector:
    def __init__(self, strategies, name=""):
        self.name = name  # For logging distinction
        self.strategies = strategies + ["INACTION"]
        self.num_arms = len(self.strategies)
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
        # New: Tracking
        self.selections = np.zeros(self.num_arms, dtype=int)  # Count of selections per arm
        self.rewards = [[] for _ in range(self.num_arms)]  # List of rewards per arm (for impact analysis)
        self.sequence = []  # Sequence of (generation, arm) tuples
    
    def select_arm(self, generation=None):
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        arm = np.argmax(samples)
        self.selections[arm] += 1  # Track selection
        if generation is not None:
            self.sequence.append((generation, arm))  # Track order
        logging.info(f"Selected {self.name} arm {arm} ({self.strategies[arm]}) in generation {generation}")
        return arm
    
    def update(self, arm, reward):
        logging.info(f"Updating {self.name} bandit arm {arm} ({self.strategies[arm]}): reward={reward}")
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1
        self.rewards[arm].append(reward)  # Track reward for impact
    
    # New: Export data for analysis
    def get_data(self):
        return {
            'strategies': self.strategies,
            'selections': self.selections.tolist(),
            'alphas': self.alphas.tolist(),
            'betas': self.betas.tolist(),
            'rewards': self.rewards,
            'sequence': self.sequence
        }

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
        valid_ratio = len(y_pred) / len(y_pred_all)
        adjusted_score = f1_macro * valid_ratio

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
            'detailed_report_string': f"Skipping metrics due to error: {e}"
        }
        return 0.0, metrics

def mutate_instruction(parent_instr, strategy, llm):
    if strategy == "INACTION":
        return parent_instr
    prompt = META_PROMPT_INSTR_4.format(strategy=strategy, parent_instr=parent_instr)
    return llm.query([prompt])[0]

def mutate_template(parent_template, template_strategy, llm, statutory_enabled, contract_enabled):
    if template_strategy == "INACTION":
        return parent_template
    prompt = META_PROMPT_TEMPLATE_3.format(strategy=template_strategy, parent_template=parent_template)
    if not statutory_enabled:
        prompt += "\nDo not include the <statutory_context> placeholder in the new template."
    if not contract_enabled:
        prompt += "\nDo not include the <contract_context> placeholder in the new template."
    return llm.query([prompt])[0]

def mutate_prompt_ga(parent, instr_selector, template_selector, llm, use_bandit_instr, use_bandit_template, statutory_enabled, contract_enabled, generation):
    if use_bandit_instr:
        instr_arm = instr_selector.select_arm(generation=generation)
    else:
        instr_arm = random.randint(0, instr_selector.num_arms - 1)
    instr_strategy = instr_selector.strategies[instr_arm]
    new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
    
    if use_bandit_template:
        template_arm = template_selector.select_arm(generation=generation)
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

def optimize_prompt(train_x, train_context, train_y, llm, generations=50, pop_size=10, train_sample_size=50, use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True, run_id=None):
    base_instr = get_base_instr(statutory_context_enabled, contract_context_enabled)
    base_template = get_base_template(statutory_context_enabled, contract_context_enabled)
    
    population = [Prompt(base_instr, base_template) for _ in range(pop_size)]  # Start with identical bases
    instr_selector = BanditSelector(INSTRUCTION_STRATEGIES_ORIGINAL, name="Instruction")
    template_selector = BanditSelector(TEMPLATE_STRATEGIES, name="Template")
    
    # New: Track improvements per strategy
    instr_improvements = [[] for _ in range(instr_selector.num_arms)]  # List of score deltas per instr arm
    template_improvements = [[] for _ in range(template_selector.num_arms)]  # Same for template
    
    best_score = 0.0
    no_improve_gens = 0
    max_no_improve = 500  # Early stopping if no improvement for 5 gens # currently set to 200 for debugging
    
    for gen in range(generations):
        logging.info(f"Starting generation {gen + 1}")
        print(f"============ Generation {gen + 1} ============")
        
        scores_and_metrics = [evaluate(p, train_x, train_context, train_y, llm, sample_size=train_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled) for p in tqdm(population, desc="Evaluating population")]
        scores = [s[0] for s in scores_and_metrics]
        # We don't need train metrics for return, just for scoring
        for p, score in zip(population, scores):
            p.score = score if not np.isnan(score) else 0.0  # Handle NaN
        population.sort(key=lambda p: p.score, reverse=True)
        print("⭐⭐ Scores:", [p.score for p in population])
        
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
        for _ in range(pop_size - len(top_k)):
            parent = random.choice(top_k)
            child = mutate_prompt_ga(parent, instr_selector, template_selector, llm, use_bandit_instr, use_bandit_template, statutory_context_enabled, contract_context_enabled, generation=gen + 1)
            child_score, _ = evaluate(child, train_x, train_context, train_y, llm, sample_size=train_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled)
            child.score = child_score
            
            parent_max = max(p.score for p in top_k)
            reward = 1 if child.score > parent_max else 0
            instr_selector.update(child.instr_arm, reward)
            template_selector.update(child.template_arm, reward)  # Same reward for both
            
            # New: Track improvement delta
            delta = child.score - parent.score  # Or use parent_max if preferred
            instr_improvements[child.instr_arm].append(delta)
            template_improvements[child.template_arm].append(delta)
            
            children.append(child)
        
        population = top_k + children
    
    best_prompt = max(population, key=lambda p: p.score)
    # Save best prompt to file
    with open("best_prompt.json", "w") as f:
        json.dump({"instr": best_prompt.instr, "template": best_prompt.template}, f)
    logging.info("Best prompt saved to best_prompt.json")
    
    logging.info(f"Best Instruction: {best_prompt.instr}")
    logging.info(f"Best Template: {best_prompt.template}")
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)
    
    # New: Export data (with run_id if multiple runs)
    instr_file = f"instr_bandit_data{run_id if run_id else ''}.json"
    template_file = f"template_bandit_data{run_id if run_id else ''}.json"
    with open(instr_file, "w") as f:
        data = instr_selector.get_data()
        data['improvements'] = instr_improvements  # Add deltas
        json.dump(data, f)
    with open(template_file, "w") as f:
        data = template_selector.get_data()
        data['improvements'] = template_improvements
        json.dump(data, f)
    logging.info(f"Bandit data exported to {instr_file} and {template_file}")
    
    return best_prompt, instr_file, template_file

def analyze_bandit_data(instr_files, template_files, bandit_type="Instruction"):  # bandit_type for labeling
    # Load and aggregate data from multiple files if list
    def aggregate_data(files):
        aggregated = {
            'strategies': None,
            'selections': None,
            'rewards': [[] for _ in range(100)],  # Assume max arms
            'improvements': [[] for _ in range(100)],
            'sequence': [],
            'alphas': None,
            'betas': None
        }
        num_runs = len(files)
        for i, file in enumerate(files):
            with open(file, 'r') as f:
                data = json.load(f)
            if aggregated['strategies'] is None:
                aggregated['strategies'] = data['strategies']
                num_arms = len(data['strategies'])
                aggregated['selections'] = np.zeros(num_arms)
                aggregated['alphas'] = np.zeros(num_arms)
                aggregated['betas'] = np.zeros(num_arms)
                aggregated['rewards'] = [[] for _ in range(num_arms)]
                aggregated['improvements'] = [[] for _ in range(num_arms)]
            aggregated['selections'] += np.array(data['selections'])
            aggregated['alphas'] += np.array(data['alphas'])
            aggregated['betas'] += np.array(data['betas'])
            for arm in range(num_arms):
                aggregated['rewards'][arm].extend(data['rewards'][arm])
                aggregated['improvements'][arm].extend(data.get('improvements', [[] for _ in range(num_arms)])[arm])
            # Offset generation for sequences to avoid overlap
            aggregated['sequence'].extend([(gen + (i * 1000), arm) for gen, arm in data['sequence']])  # Arbitrary offset
        # Average alphas/betas
        aggregated['alphas'] /= num_runs
        aggregated['betas'] /= num_runs
        aggregated['selections'] /= num_runs  # Average selections per run
        return aggregated, num_runs

    # Analyze for one bandit (instr or template)
    data, num_runs = aggregate_data(instr_files if bandit_type == "Instruction" else template_files)
    strategies = data['strategies']
    num_arms = len(strategies)
    total_selections = sum(data['selections'])

    # 1. Frequency: Selections
    freq_df = pd.DataFrame({
        'Strategy': strategies,
        'Avg Selections': data['selections'],
        'Percentage': (data['selections'] / total_selections * 100) if total_selections > 0 else [0] * num_arms
    }).sort_values('Avg Selections', ascending=False)
    print(f"\n=== {bandit_type} Strategies by Frequency (Chosen More Often) Across {num_runs} Runs ===")
    print(freq_df.to_string(index=False))

    # Chart: Bar for selections
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_arms), data['selections'], tick_label=[s[:20] + '...' for s in strategies])  # Shorten labels
    plt.title(f'Average Selections per Strategy ({bandit_type})')
    plt.xlabel('Strategy')
    plt.ylabel('Avg Selections')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{bandit_type.lower()}_selections.png")
    plt.show()  # Shows if in interactive env

    # 2. Impact: Success rate and avg delta
    success_rates = []
    avg_deltas = []
    for arm in range(num_arms):
        rewards = data['rewards'][arm]
        improvements = data['improvements'][arm]
        success_rate = np.mean(rewards) if rewards else 0
        avg_delta = np.mean(improvements) if improvements else 0
        success_rates.append(success_rate)
        avg_deltas.append(avg_delta)
    impact_df = pd.DataFrame({
        'Strategy': strategies,
        'Success Rate': success_rates,
        'Avg Improvement Delta': avg_deltas
    }).sort_values('Avg Improvement Delta', ascending=False)
    print(f"\n=== {bandit_type} Strategies by Impact Across {num_runs} Runs ===")
    print(impact_df.to_string(index=False))

    # Chart: Bar for avg deltas
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_arms), avg_deltas, tick_label=[s[:20] + '...' for s in strategies])
    plt.title(f'Average Improvement Delta per Strategy ({bandit_type})')
    plt.xlabel('Strategy')
    plt.ylabel('Avg Delta')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{bandit_type.lower()}_improvements.png")
    plt.show()

    # 3. Preferences: Mean probability from beta dist (alpha / (alpha + beta))
    means = [data['alphas'][arm] / (data['alphas'][arm] + data['betas'][arm]) if data['alphas'][arm] + data['betas'][arm] > 0 else 0 for arm in range(num_arms)]
    pref_df = pd.DataFrame({
        'Strategy': strategies,
        'Preference Score': means
    }).sort_values('Preference Score', ascending=False)
    print(f"\n=== {bandit_type} Strategies by Preference (Chosen Over Others) Across {num_runs} Runs ===")
    print(pref_df.to_string(index=False))

    # 4. Order: Sequence analysis
    if data['sequence']:
        # Common first/last strategies
        sequences = [arm for _, arm in data['sequence']]
        first_strat = strategies[np.argmax(np.bincount([arm for gen, arm in data['sequence'] if gen % 1000 == 1]))]  # Approx early
        last_strat = strategies[np.argmax(np.bincount([arm for gen, arm in data['sequence'] if gen % 1000 > 900]))]  # Approx late
        # Transitions: Count arm_i -> arm_j
        transitions = np.zeros((num_arms, num_arms))
        for i in range(len(sequences) - 1):
            transitions[sequences[i], sequences[i+1]] += 1
        transitions /= np.sum(transitions, axis=1, keepdims=True) + 1e-5  # Normalize to probs
        print(f"\n=== {bandit_type} Strategy Order Summary Across {num_runs} Runs ===")
        print(f"Most common early strategy: {first_strat}")
        print(f"Most common late strategy: {last_strat}")
        print("Top Transitions (Prob > 0.1):")
        for i in range(num_arms):
            for j in range(num_arms):
                if transitions[i, j] > 0.1:
                    print(f"{strategies[i][:20]}... -> {strategies[j][:20]}...: {transitions[i, j]:.2f}")
    else:
        print(f"\n=== {bandit_type} Strategy Order Summary: No sequences recorded. ===")

# ========== Main ============
def main(generations=20, pop_size=8, train_sample_size=10, test_sample_size=100, model_name="google/gemini-2.5-flash-lite-preview-06-17", use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True, num_runs=1):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = Data.load(os.path.join(base_dir, "train_unskewed.tsv"))
    test_data = Data.load(os.path.join(base_dir, "test.tsv"))
    
    llm = OpenRouterLLM(model_name)
    
    instr_files = []
    template_files = []
    best_prompts = []
    for run in range(1, num_runs + 1):
        print(f"\n===== Run {run}/{num_runs} =====")
        random.seed(42 + run)  # Different seed per run
        np.random.seed(42 + run)
        best_prompt, instr_file, template_file = optimize_prompt(
            train_data.get_x(), train_data.get_context(), train_data.get_y(), llm, generations, pop_size, train_sample_size,
            use_bandit_instr, use_bandit_template, statutory_context_enabled, contract_context_enabled, run_id=f"_run{run}"
        )
        best_prompts.append(best_prompt)
        instr_files.append(instr_file)
        template_files.append(template_file)
    
    # Analyze aggregated data
    print("\n===== Strategy Analysis (Aggregated Across Runs) =====")
    analyze_bandit_data(instr_files, template_files, bandit_type="Instruction")
    analyze_bandit_data(instr_files, template_files, bandit_type="Template")
    
    # Evaluate best prompt from last run on test set (or average if multiple)
    best_prompt = best_prompts[-1]  # Use last for simplicity
    print("\nRunning on test set...")
    test_f1, test_metrics = evaluate(best_prompt, test_data.get_x(), test_data.get_context(), test_data.get_y(), llm, sample_size=test_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled)
    print(f"Test Adjusted Macro F1: {test_f1:.4f}")
    logging.info(f"Test Adjusted Macro F1: {test_f1:.4f}")

    result = {
        'best_instruction': best_prompt.instr,
        'best_template': best_prompt.template,
        'test_metrics': test_metrics
    }
    return result

if __name__ == "__main__":
    main(num_runs=1)  # Set to >1 for multiple runs and aggregated analysis