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

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ========== Strategy List from APET Paper ============

strategies_list = [
    "Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.",
    "Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding \"Let's think step-by-step\".",
    "Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.",
    "Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.",
    "At the end of the prompt, add a phrase that evokes a strong emotion.",
    "For a given prompt, add a phrase such as \"Read the question again\" that instructs the Large Language Models to reread the question before generating an answer.",
    "Clearly define the desired style in the given prompt.",
    "Add a phrase that instructs the model to rephrase the question before responding.",
    "Make the prompt more specific.",
    "Add phrases that instruct the model to remove opinionated content.",
    "Shorten long prompts to essential parts.",
    "Edit the prompt instruction to invoke legal reasoning for problem solving: 1) State the goal of determining the unfairness of a clause. 2) Give a detailed definition of what could be considered an unfair clause. 3) compare the given sentence with the definition to estimate which parts of the sentence falls under that definition. 4) make a final determination based on the comparison."
]

# ========== Utility Classes ============

class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
        self.score = np.nan
        self.select_arm = np.nan

    def get_user(self):
        return self.instr

    def join_input(self, text):
        return self.template.replace("<prompt>", self.instr).replace("<q>", text)


class Data:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def load(path, delimiter="\t"):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip().split(delimiter) for line in f.readlines() if line.strip()]
        return Data(lines)

    def get_x(self):
        return [line[4] for line in self.dataset]  # 'text' column (index 4)

    def get_y(self):
        return [line[2] for line in self.dataset]  # 'label' column (index 2)


# ========== LLM Interface (OpenAI 1.x API) ============

class OpenRouterLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, prompts, temperature=0.7, max_tokens=256):
        outputs = []
        for prompt in prompts:
            while True:
                try:
                    client = openai.OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # time.sleep(0.5)  
                    outputs.append(response.choices[0].message.content.strip())
                    print("Prompt:", prompt)
                    print("Response:", outputs[-1])
                    break
                except Exception as e:
                    print("Retrying due to error:", e)
                    time.sleep(1)
        return outputs


# ========== Evaluator ============

def evaluate(prompt_obj, data_x, data_y, llm, batch_size=5):
    outputs = []
    for i in range(0, len(data_x), batch_size):
        print("Processing batch:", i // batch_size + 1)
        batch = data_x[i:i+batch_size]
        formatted = [prompt_obj.join_input(x) for x in batch]
        batch_outputs = llm.query(formatted, temperature=0.0)
        outputs.extend(batch_outputs)

    def clean_pred(pred):
        # Accept only '0' or '1', else default to '0'
        return '1' if str(pred).strip() == '1' else '0'

    cleaned_outputs = [extract_answer(o) for o in outputs]
    y_true = [str(label).strip() for label in data_y]
    y_pred = [clean_pred(pred) for pred in cleaned_outputs]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    support = {k: v['support'] for k, v in report.items() if k in ['0', '1']}

    print(f"Sample size: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Micro F1: {f1_micro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Support: {support}")
    print("Detailed classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Unique y_true:", set(y_true))
    print("Unique y_pred:", set(y_pred))

    return accuracy


def extract_answer(output):
    match = re.search(r"\b([01])\b", output)
    if match:
        return match.group(1)
    return output.strip()


# ========== OPTS-TS with GA (EvoPromptGA-OPTS-TS) ============

def mutate_prompt_ga(parent, strategy, llm):
    prompt = (
        f"You are an expert prompt engineer applying the following transformation strategy to improve a prompt for a classification task. It is important that responses at all times only consist '0' for fair or '1' for unfair.\n"
        f"Strategy: {strategy}\n"
        f"Original Prompt: {parent.instr}\n"
        f"New Prompt:"
    )
    new_instr = llm.query([prompt])[0]
    template_prompt = (
        f"Generate a prompt template using placeholders <q> for the input question and <prompt> for the instruction. It is important that responses at all times only consist '0' for fair or '1' for unfair.\n"
        f"Instruction: {new_instr}\n"
        f"Prompt Template:"
    )
    template = llm.query([template_prompt])[0]
    return Prompt(new_instr, template)


def optimize_prompt(train_x, train_y, llm, generations=5, pop_size=6):
    base_instr = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1). Respond only with '0' or '1'."
    base_template = "Clause: <q>\nA: <prompt>\n"
    population = [Prompt(base_instr, base_template)]

    for _ in range(pop_size - 1):
        strategy = random.choice(strategies_list)
        population.append(mutate_prompt_ga(population[0], strategy, llm))

    for gen in range(generations):
        print(f"Generation {gen + 1}")
        scores = [evaluate(p, train_x, train_y, llm) for p in population]
        for i, p in enumerate(population):
            p.score = scores[i]
        population = sorted(population, key=lambda p: p.score, reverse=True)

        top_k = population[:pop_size//2]
        children = []
        for _ in range(pop_size - len(top_k)):
            parent = random.choice(top_k)
            strategy = random.choice(strategies_list)
            children.append(mutate_prompt_ga(parent, strategy, llm))

        population = top_k + children

    best_prompt = max(population, key=lambda p: p.score)
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)
    return best_prompt


# ========== Main Pipeline ============

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.tsv")
    test_path = os.path.join(base_dir, "test.tsv")

    train_data = Data.load(train_path)
    test_data = Data.load(test_path)

    sample_indices = random.sample(range(len(train_data.dataset)), 5) #5
    sampled_dataset = [train_data.dataset[i] for i in sample_indices]
    sampled_train = Data(sampled_dataset)

    llm = OpenRouterLLM("google/gemini-2.0-flash-001")

    best_prompt = optimize_prompt(
        sampled_train.get_x(),
        sampled_train.get_y(),
        llm=llm,
        generations=3, #5
        pop_size=3, #6
    )

    print("\nRunning on test set...")
    sample_size = 10
    test_indices = random.sample(range(len(test_data.dataset)), min(sample_size, len(test_data.dataset)))
    sampled_test = Data([test_data.dataset[i] for i in test_indices])
    test_accuracy = evaluate(best_prompt, sampled_test.get_x(), sampled_test.get_y(), llm)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
