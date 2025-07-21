import os
import random
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import openai
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import csv
import scipy.stats as stats

csv.field_size_limit(10_000_000)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ========== Constants ============
INSTRUCTION_STRATEGIES = [
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
]

TEMPLATE_STRATEGIES = [
    "Enhance the description of relationships between template elements, for example explaining how the statutory context provides legal foundations, the contract context offers specific background, the instruction guides the process, and the clause is the target for classification.",
    "Reorder the template elements to optimize logical flow, for example presenting the statutory context first, followed by contract context, instruction, and clause or another arrangement that could be better.",
    "Incorporate separators, delimiters, or formatting emphasis (e.g., bold, italics) to improve readability and highlight key sections of the template.",
    "Experimentally completely omit one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially simplifying or enriching it while maintaining the classification task's integrity with <instruction> and <clause>.",
    "Experimentally re-add one or more of the placeholders <statutory_context> and <contract_context> to refine the template, potentially enriching it while maintaining the classification task's integrity with <instruction> and <clause>."
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

# ========== Classes ============
class BanditSelector:
    def __init__(self, strategies):
        self.strategies = strategies + ["INACTION"]
        self.num_arms = len(self.strategies)
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)
    
    def select_arm(self):
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
        self.score = np.nan
        self.select_arm = None
    
    def join_input(self, text, context):
        return self.template.replace("<instruction>", self.instr).replace("<clause>", text).replace("<contract_context>", context).replace("<statutory_context>", PLACEHOLDER_STATUTORY_CONTEXT)

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
                    outputs.append(response.choices[0].message.content.strip())
                    break
                except Exception as e:
                    print(f"Retrying due to error: {e}")
                    time.sleep(2 ** retries)
                    retries += 1
        return outputs

# ========== Functions ============
def extract_answer(output):
    output = output.strip()
    if output in ['0', '1']:
        return output
    matches = re.findall(r'(?<!\d)[01](?!\d)', output)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"⚠️ Multiple digits found, taking last: '{output}'")
        return matches[-1]
    print(f"⚠️ Invalid answer: '{output}'")
    return 'invalid'

def evaluate(prompt_obj, data_x, data_context, data_y, llm, batch_size=20, sample_size=10):
    n = len(data_x)
    indices = random.sample(range(n), min(sample_size, n))
    eval_x = [data_x[i] for i in indices]
    eval_context = [data_context[i] for i in indices]
    eval_y = [data_y[i] for i in indices]

    outputs = []
    for i in range(0, len(eval_x), batch_size):
        batch_x = eval_x[i:i+batch_size]
        batch_context = eval_context[i:i+batch_size]
        formatted = [prompt_obj.join_input(x, c) for x, c in zip(batch_x, batch_context)]
        outputs.extend(llm.query(formatted, temperature=0.0))

    cleaned_outputs = [extract_answer(o) for o in outputs]
    y_true = [str(y).strip() for y in eval_y]
    y_pred = [p if p in ['0', '1'] else 'invalid' for p in cleaned_outputs]

    valid_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_true)) if pred in ['0', '1'] and true in ['0', '1']]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]

    if not y_pred_valid:
        print("❌ No valid predictions. Score: 0.")
        return 0.0

    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
    print(f"Eval F1-Macro: {f1_macro:.4f} (Valid: {len(y_pred_valid)}/{len(y_pred)})")
    return f1_macro

def mutate_instruction(parent_instr, strategy, llm):
    if strategy == "INACTION":
        return parent_instr
    prompt = META_PROMPT_INSTR.format(strategy=strategy, parent_instr=parent_instr)
    return llm.query([prompt])[0]

def mutate_template(parent_template, template_strategy, llm):
    prompt = (
        f"You are an expert prompt engineer applying: {template_strategy}\n"
        f"Ensure placeholders: <instruction>, <clause>, optionally <contract_context>, <statutory_context>.\n"
        f"Responses must be '0' or '1'.\n"
        f"Original Template: {parent_template}\nNew Template:"
    )
    return llm.query([prompt])[0]

def mutate_prompt_ga(parent, bandit_selector, llm):
    arm = bandit_selector.select_arm()
    instr_strategy = bandit_selector.strategies[arm]
    new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
    
    template_strategy = random.choice(TEMPLATE_STRATEGIES)
    new_template = mutate_template(parent.template, template_strategy, llm)
    
    child = Prompt(new_instr, new_template)
    child.select_arm = arm
    return child

def optimize_prompt(train_x, train_context, train_y, llm, generations=10, pop_size=4):
    base_instr = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1) using the context for better understanding. Respond only with '0' or '1'."
    base_template = "Instruction: <instruction>\nClause: <clause>\nStatutory Context: <statutory_context>\nContract Context: <contract_context>"
    
    population = [Prompt(base_instr, base_template) for _ in range(pop_size)]
    bandit_selector = BanditSelector(INSTRUCTION_STRATEGIES)
    
    for gen in range(generations):
        print(f"Generation {gen + 1}")
        scores = [evaluate(p, train_x, train_context, train_y, llm) for p in tqdm(population)]
        for p, score in zip(population, scores):
            p.score = score
        population.sort(key=lambda p: p.score, reverse=True)
        print("Scores:", [p.score for p in population])
        
        top_k = population[:pop_size // 2]
        children = []
        for _ in range(pop_size - len(top_k)):
            parent = random.choice(top_k)
            child = mutate_prompt_ga(parent, bandit_selector, llm)
            child.score = evaluate(child, train_x, train_context, train_y, llm)
            
            parent_max = max(p.score for p in top_k)
            reward = 1 if child.score > parent_max else 0
            bandit_selector.update(child.select_arm, reward)
            children.append(child)
        
        population = top_k + children
    
    best_prompt = max(population, key=lambda p: p.score)
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)
    return best_prompt

# ========== Main ============
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = Data.load(os.path.join(base_dir, "train_unskewed.tsv"))
    test_data = Data.load(os.path.join(base_dir, "test.tsv"))
    
    llm = OpenRouterLLM("google/gemini-2.5-flash-lite-preview-06-17")
    best_prompt = optimize_prompt(train_data.get_x(), train_data.get_context(), train_data.get_y(), llm)
    
    print("\nRunning on test set...")
    test_f1 = evaluate(best_prompt, test_data.get_x(), test_data.get_context(), test_data.get_y(), llm, sample_size=100)
    print(f"Test Macro F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()