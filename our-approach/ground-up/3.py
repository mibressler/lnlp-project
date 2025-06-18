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

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_key = api_key

# ========== Strategy List from APET Paper ============

strategies_list = [
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
    "If a given prompt has long instructions, make it shorter by condensing it to only the essential parts.",
]

# ========== Utility Classes ============

class Prompt:
    def __init__(self, instr, template="Q: <q>\nA: <prompt>\n"):
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
        return [line[0] for line in self.dataset]

    def get_y(self):
        return [line[1] for line in self.dataset]


# ========== LLM Interface ============

class OpenRouterLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, prompts, temperature=0.7, max_tokens=256):
        outputs = []
        for prompt in prompts:
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    outputs.append(response["choices"][0]["message"]["content"].strip())
                    break
                except Exception as e:
                    print("Retrying due to error:", e)
                    time.sleep(1)
        return outputs


# ========== Evaluator ============

def evaluate(prompt_obj, data_x, data_y, llm, batch_size=5):
    outputs = []
    for i in range(0, len(data_x), batch_size):
        batch = data_x[i:i+batch_size]
        formatted = [prompt_obj.join_input(x) for x in batch]
        batch_outputs = llm.query(formatted, temperature=0.0)
        outputs.extend(batch_outputs)

    cleaned_outputs = [extract_answer(o) for o in outputs]
    accuracy = np.mean([str(int(pred == label)) for pred, label in zip(cleaned_outputs, data_y)]).astype(float)
    return accuracy


def extract_answer(output):
    match = re.search(r"(?<=the answer is )(\d)", output)
    if match:
        return match.group(1)
    return output.strip()


# ========== OPTS-TS + GA Simplified ============

def paraphrase_instruction(instr, llm):
    prompt = f"Rewrite the following prompt to mean the same thing: \"{instr}\""
    return llm.query([prompt])[0]


def mutate_prompt_ga(parent1, parent2, llm):
    strategy = random.choice(strategies_list)
    prompt = (
        f"You're an AI prompt engineer. Use the following strategy to create a new prompt.\n"
        f"Strategy: {strategy}\n"
        f"Instruction 1: {parent1}\n"
        f"Instruction 2: {parent2}\n"
        f"New Prompt:"
    )
    return llm.query([prompt])[0]


def optimize_prompt(train_x, train_y, llm, generations=5, pop_size=10):
    base_instr = "Classify the following sentence as fair (0) or unfair (1). Respond with 'the answer is 0' or 'the answer is 1'."
    population = [Prompt(base_instr)]
    for _ in range((pop_size - 1)):
        paraphrased = paraphrase_instruction(base_instr, llm)
        population.append(Prompt(paraphrased))

    for gen in range(generations):
        print(f"Generation {gen + 1}")
        scores = [evaluate(p, train_x, train_y, llm) for p in population]
        print("Scores:", scores)
        top = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)[:pop_size//2]
        parents = [p for p, _ in top]
        children = []
        for _ in range(pop_size - len(parents)):
            p1, p2 = random.sample(parents, 2)
            child_instr = mutate_prompt_ga(p1.get_user(), p2.get_user(), llm)
            children.append(Prompt(child_instr))
        population = parents + children

    best_prompt = max(population, key=lambda p: evaluate(p, train_x, train_y, llm))
    print("Best Prompt:", best_prompt.get_user())
    return best_prompt


# ========== Main Pipeline ============

def main():
    train_data = Data.load("train.tsv")
    test_data = Data.load("test.tsv")

    # Sample 20 from train for quick tuning
    sample_indices = random.sample(range(len(train_data.dataset)), 20)
    sampled_dataset = [train_data.dataset[i] for i in sample_indices]
    sampled_train = Data(sampled_dataset)

    llm = OpenRouterLLM("mistralai/mistral-7b-instruct")

    best_prompt = optimize_prompt(
        sampled_train.get_x(),
        sampled_train.get_y(),
        llm=llm,
        generations=5,
        pop_size=6,
    )

    print("\nRunning on test set...")
    test_accuracy = evaluate(best_prompt, test_data.get_x(), test_data.get_y(), llm)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()