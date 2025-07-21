import os
import random
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import openai
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import csv
import scipy.stats as stats  # For beta distribution in TS

csv.field_size_limit(10_000_000)

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ========== Strategy Lists (Use Concise per Paper) ============
instruction_strategies = [...]  # Your original list (omitted for brevity)
instruction_strategies_concise = [  # Use this for efficiency
    # Your concise list here (omitted for brevity)
]

template_strategies = [  # Kept as-is
    # Your list here
]

placeholder_statutory_context = (  # Kept as-is
    # Your context here
)

# ========== Meta-Prompts (Inspired by Paper's Table 6) ============
META_PROMPT_INSTR = """
Imagine yourself as an expert in prompting techniques for LLMs. Your expertise is broad and deep. Your job is to reformulate instructions with precision, optimizing for accurate responses in a legal classification task. The reformulated instruction MUST ensure responses are strictly '0' for fair or '1' for unfair.

Available technique: {strategy}

Reformulate the below instruction using the technique. Include ALL original information. ONLY return the reformulated instruction.

Original Instruction: {parent_instr}
"""

META_PROMPT_TEMPLATE = """  # Similar for templates, if extending TS
# ... (omitted; add if needed)
"""

# ========== Bandit Selector for OPTS(TS) ============
class BanditSelector:
    def __init__(self, strategies):
        self.strategies = strategies + ["INACTION"]  # Add inaction arm
        self.num_arms = len(self.strategies)
        self.alphas = np.ones(self.num_arms)  # Beta priors (alpha=1, beta=1)
        self.betas = np.ones(self.num_arms)
    
    def select_arm(self):
        samples = [stats.beta.rvs(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

# ========== Utility Classes (Minor tweaks) ============
class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
        self.score = np.nan

    def join_input(self, text, context):
        return self.template.replace("<instruction>", self.instr).replace("<clause>", text).replace("<contract_context>", context).replace("<statutory_context>", placeholder_statutory_context)

# Data class unchanged

# ========== LLM Interface (Added backoff) ============
class OpenRouterLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, prompts, temperature=0.7, max_tokens=256):
        outputs = []
        for prompt in prompts:
            retries = 0
            while retries < 3:  # Improved retry with exponential backoff
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
                    time.sleep(2 ** retries)  # Exponential backoff
                    retries += 1
        return outputs

# ========== Evaluator (Increased sample_size, log invalids) ============
def evaluate(prompt_obj, data_x, data_context, data_y, llm, batch_size=20, sample_size=50):  # Increased to 50 for stability
    # ... (your code, with added logging)
    # In the end:
    if not y_pred:
        print("❌ No valid predictions. Falling back to 0.")
        return 0.0
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Eval F1-Macro: {f1_macro:.4f}")
    return f1_macro

# extract_answer unchanged

# ========== OPTS-TS with GA (EvoPrompt-OPTS Inspired) ============
def mutate_instruction(parent_instr, strategy, llm):
    if strategy == "INACTION":
        return parent_instr
    prompt = META_PROMPT_INSTR.format(strategy=strategy, parent_instr=parent_instr)
    return llm.query([prompt])[0]

def mutate_template(parent_template, template_strategy, llm):  # Kept random; extend TS if desired
    prompt = f"You are an expert prompt engineer applying: {template_strategy}\nOriginal Template: {parent_template}\nNew Template:"  # Simplified; use meta if needed
    return llm.query([prompt])[0]

def mutate_prompt_ga(parent, bandit_selector, llm):
    # Select strategy via TS
    arm = bandit_selector.select_arm()
    instr_strategy = bandit_selector.strategies[arm]
    new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
    
    # Random template mutation (as before)
    template_strategy = random.choice(template_strategies)
    new_template = mutate_template(parent.template, template_strategy, llm)
    
    child = Prompt(new_instr, new_template)
    child.select_arm = arm  # Track for update
    return child

def optimize_prompt(train_x, train_context, train_y, llm, generations=50, pop_size=10):  # Paper's defaults
    base_instr = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1) using the context for better understanding. Respond only with '0' or '1'."
    base_template = "Instruction: <instruction>\nClause: <clause>\nStatutory Context: <statutory_context>\nContract Context: <contract_context>"
    
    population = [Prompt(base_instr, base_template) for _ in range(pop_size)]
    bandit_selector = BanditSelector(instruction_strategies_concise)  # OPTS(TS)
    
    for gen in range(generations):
        print(f"Generation {gen + 1}")
        scores = [evaluate(p, train_x, train_context, train_y, llm, sample_size=50) for p in tqdm(population)]
        for i, p in enumerate(population):
            p.score = scores[i]
        population = sorted(population, key=lambda p: p.score, reverse=True)
        print("Scores:", [p.score for p in population])
        
        top_k = population[:pop_size // 2]
        children = []
        for _ in range(pop_size - len(top_k)):
            parent = random.choice(top_k)
            child = mutate_prompt_ga(parent, bandit_selector, llm)
            child.score = evaluate(child, train_x, train_context, train_y, llm, sample_size=50)
            
            # Update bandit (reward: 1 if child improves over max parent scores)
            parent_max = max(p.score for p in top_k)
            reward = 1 if child.score > parent_max else 0
            bandit_selector.update(child.select_arm, reward)
            children.append(child)
        
        population = top_k + children
    
    best_prompt = max(population, key=lambda p: p.score)
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)
    return best_prompt

# ========== Main Pipeline (Unchanged) ============
def main():
    # ... (your paths and data loading)
    llm = OpenRouterLLM("google/gemini-2.5-flash-lite-preview-06-17")
    best_prompt = optimize_prompt(
        sampled_train.get_x(),
        sampled_train.get_context(),
        sampled_train.get_y(),
        llm=llm,
        generations=50,
        pop_size=10,
    )
    # Test evaluation unchanged

if __name__ == "__main__":
    main()