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
from datetime import datetime

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
    
    def select_arm(self):
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        logging.info(f"Updating {self.name} bandit arm {arm} ({self.strategies[arm]}): reward={reward}")
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

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
                    
                    
                    #print("\n=== SENT ===")
                    #print(prompt)
                    #print("\n=== RECEIVED ===")
                    #print(output)
                    #print("\n=============\n")
                    
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
            'detailed_report_string': f"Skipping metrics due to error: {e}"
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

def optimize_prompt(train_x, train_context, train_y, llm, generations=50, pop_size=10, train_sample_size=50, use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True, timestamp=""):
    base_instr = get_base_instr(statutory_context_enabled, contract_context_enabled)
    base_template = get_base_template(statutory_context_enabled, contract_context_enabled)
    
    population = [Prompt(base_instr, base_template) for _ in range(pop_size)]  # Start with identical bases
    instr_selector = BanditSelector(INSTRUCTION_STRATEGIES_ORIGINAL, name="Instruction")
    template_selector = BanditSelector(TEMPLATE_STRATEGIES, name="Template")
    
    best_score = 0.0
    no_improve_gens = 0
    max_no_improve = 500  # Early stopping if no improvement for 5 gens # currently set to 200 for debugging
    
    prompt_id_counter = 0
    score_history = {}
    for p in population:
        p.id = prompt_id_counter
        prompt_id_counter += 1
    
    for gen in range(generations):
        logging.info(f"Starting generation {gen + 1}")
        print(f"============ Generation {gen + 1} ============")
        
        scores_and_metrics = [evaluate(p, train_x, train_context, train_y, llm, sample_size=train_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled) for p in tqdm(population, desc="Evaluating population")]
        scores = [s[0] for s in scores_and_metrics]
        # We don't need train metrics for return, just for scoring
        for p, score in zip(population, scores):
            p.score = score if not np.isnan(score) else 0.0  # Handle NaN
        
        # Append scores for current population
        for p in population:
            if not hasattr(p, 'id'):
                p.id = prompt_id_counter
                prompt_id_counter += 1
            if p.id not in score_history:
                score_history[p.id] = []
            score_history[p.id].append((gen + 1, p.score))
        
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
            child = mutate_prompt_ga(parent, instr_selector, template_selector, llm, use_bandit_instr, use_bandit_template, statutory_context_enabled, contract_context_enabled)
            child.id = prompt_id_counter
            prompt_id_counter += 1
            child_score, _ = evaluate(child, train_x, train_context, train_y, llm, sample_size=train_sample_size, statutory_enabled=statutory_context_enabled, contract_enabled=contract_context_enabled)
            child.score = child_score if not np.isnan(child_score) else 0.0
            # Append score for child
            if child.id not in score_history:
                score_history[child.id] = []
            score_history[child.id].append((gen + 1, child.score))
            
            parent_max = max(p.score for p in top_k)
            reward = 1 if child.score > parent_max else 0
            instr_selector.update(child.instr_arm, reward)
            template_selector.update(child.template_arm, reward)  # Same reward for both
            children.append(child)
        
        population = top_k + children
    
    best_prompt = max(population, key=lambda p: p.score)
    # Save best prompt to file
    with open(f"best_prompt_{timestamp}.json", "w") as f:
        json.dump({"instr": best_prompt.instr, "template": best_prompt.template}, f)
    logging.info(f"Best prompt saved to best_prompt_{timestamp}.json")
    
    logging.info(f"Best Instruction: {best_prompt.instr}")
    logging.info(f"Best Template: {best_prompt.template}")
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)

    # Simple analysis of bandit selectors
    print("\n=== Strategy Selection Analysis ===")
    print("Instruction Bandit:")
    for i, strat in enumerate(instr_selector.strategies):
        total_pulls = instr_selector.alphas[i] + instr_selector.betas[i] - 2
        successes = instr_selector.alphas[i] - 1
        mean_prob = instr_selector.alphas[i] / (instr_selector.alphas[i] + instr_selector.betas[i])
        print(f"Strategy {i} ({strat[:50]}...): Pulls={total_pulls}, Successes={successes}, Est. Success Prob={mean_prob:.4f}")

    print("\nTemplate Bandit:")
    for i, strat in enumerate(template_selector.strategies):
        total_pulls = template_selector.alphas[i] + template_selector.betas[i] - 2
        successes = template_selector.alphas[i] - 1
        mean_prob = template_selector.alphas[i] / (template_selector.alphas[i] + template_selector.betas[i])
        print(f"Strategy {i} ({strat[:50]}...): Pulls={total_pulls}, Successes={successes}, Est. Success Prob={mean_prob:.4f}")
    
    # Generate PNG graphs for slidedeck
    import matplotlib.pyplot as plt

    def generate_strategy_graph(selector, title, filename):
        data = []
        for i, strat in enumerate(selector.strategies):
            total_pulls = selector.alphas[i] + selector.betas[i] - 2
            successes = selector.alphas[i] - 1
            mean_prob = selector.alphas[i] / (selector.alphas[i] + selector.betas[i])
            data.append((strat[:50], total_pulls, successes, mean_prob))
        
        # Sort by Est. Success Prob descending
        data.sort(key=lambda x: x[3], reverse=True)
        
        labels = [d[0] for d in data]
        pulls = [d[1] for d in data]
        successes = [d[2] for d in data]
        probs = [d[3] for d in data]
        
        x = np.arange(len(labels))  # Label locations
        width = 0.25  # Bar width
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.bar(x - width, pulls, width, label='Pulls', color='tab:blue')
        ax1.bar(x, successes, width, label='Successes', color='tab:green')
        ax1.set_ylabel('Counts', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(x + width/2, probs, label='Est. Success Prob', color='tab:red', marker='o', linewidth=2)
        ax2.set_ylabel('Est. Success Prob', fontsize=12)
        ax2.legend(loc='upper right')
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved graph to {filename}")

    generate_strategy_graph(instr_selector, 'Instruction Strategy Impact', f'instruction_strategy_impact_{timestamp}.png')
    generate_strategy_graph(template_selector, 'Template Strategy Impact', f'template_strategy_impact_{timestamp}.png')
    
    # Generate PNG for incremental improvement in adjusted macro F1 over generations
    from collections import defaultdict
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute max per gen
    max_per_gen = defaultdict(float)
    all_gens = set()
    for pid, data in score_history.items():
        for g, s in data:
            all_gens.add(g)
            if s > max_per_gen[g]:
                max_per_gen[g] = s
    
    gens_sorted = sorted(all_gens)
    max_scores = [max_per_gen[g] for g in gens_sorted]
    ax.plot(gens_sorted, max_scores, color='red', linewidth=3, label='Best per Generation')
    
    # Plot individual lines (only those lasting more than 1 generation to reduce clutter)
    num_lines = len(score_history)
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))
    for i, (pid, data) in enumerate(score_history.items()):
        if len(data) > 1:
            data = sorted(data)
            gens, scores = zip(*data)
            ax.plot(gens, scores, color=colors[i], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Adjusted Macro F1')
    ax.set_title('Evolution of Prompt Scores over Generations')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'prompt_evolution_{timestamp}.png')
    plt.close()
    print(f"Saved prompt evolution graph to prompt_evolution_{timestamp}.png")
    
    return best_prompt

# ========== Main ============
def main(generations=20, pop_size=8, train_sample_size=10, test_sample_size=100, model_name="google/gemini-2.5-flash-lite-preview-06-17", use_bandit_instr=True, use_bandit_template=True, statutory_context_enabled=True, contract_context_enabled=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = Data.load(os.path.join(base_dir, "train_unskewed.tsv"))
    test_data = Data.load(os.path.join(base_dir, "test.tsv"))
    
    llm = OpenRouterLLM(model_name)
    best_prompt = optimize_prompt(train_data.get_x(), train_data.get_context(), train_data.get_y(), llm, generations, pop_size, train_sample_size, use_bandit_instr, use_bandit_template, statutory_context_enabled, contract_context_enabled, timestamp)
    
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
    main()