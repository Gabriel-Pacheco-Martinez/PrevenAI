"""
This script expands a small set of seed JSON templates into a larger dataset (~1,000 prompts)
using LangChain and a language model. 

Author: Gabriel Pacheco, 2025
"""

# General
import json
import random
from pathlib import Path
from typing import List, Dict

# LLM
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI  # Or your LLM class
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate


# ----------------------------
# INSTALLATION
# ----------------------------
# pip install langchain openai tiktoken tqdm jsonlines
# Replace `openai` with your LLM provider if using Ollama, Qwen, etc.
# Ensure you have API keys configured for your LLM.

# ----------------------------
# IMPORTS
# ----------------------------




# ----------------------------
# CONFIGURATION
# ----------------------------
SEED_FILE = "seed_templates.json"  # 20 seed prompts
OUTPUT_FILE = "expanded_templates.json"  # Final expanded JSON
EXPANSION_FACTOR = 50  # Each seed generates 50 variations -> 20*50 = 1000
RANDOM_SEED = 42  # Reproducibility

# LLM Settings
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Change to your fine-tuning model if needed
    temperature=0.7,
    max_tokens=300
)

random.seed(RANDOM_SEED)

# ----------------------------
# LOAD SEED TEMPLATES
# ----------------------------
def load_seed_templates(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# EXPANSION FUNCTION
# ----------------------------
def expand_template(template: Dict, num_variations: int = 50) -> List[Dict]:
    """
    Generates multiple prompt variations from a seed template using an LLM.
    """
    expanded = []
    prompt_text = f"""
    You are a professional preventative health assistant. 
    Generate {num_variations} diverse, professional, and self-contained variations 
    of the following Q&A template. Keep the same topic and tone. 
    Do not give personalized advice. Keep answers concise.

    TEMPLATE:
    Prompt: {template['prompt']}
    Response: {template['response']}
    Topic: {template['topic']}

    Output format: JSON array with keys "prompt", "response", "topic".
    """

    # Call LLM to generate expanded templates
    result = llm([{"role": "user", "content": prompt_text}])
    
    # Parse output (expecting JSON array)
    try:
        variations = json.loads(result.content)
        # Ensure topic is consistent
        for v in variations:
            v['topic'] = template['topic']
        expanded.extend(variations)
    except Exception as e:
        print(f"Error parsing JSON for template '{template['prompt']}': {e}")
    
    return expanded

# ----------------------------
# OPTIONAL FILTERING
# ----------------------------
def filter_templates(templates: List[Dict]) -> List[Dict]:
    """
    Optional: filter out duplicates, overly long responses, or any unsafe content.
    """
    unique_prompts = set()
    filtered = []
    for t in templates:
        prompt_text = t['prompt'].strip()
        if prompt_text not in unique_prompts and len(t['response']) < 300:
            filtered.append(t)
            unique_prompts.add(prompt_text)
    return filtered

# ======
# MAIN: entry point to expand json prompts
def main():
    seed_templates = load_seed_templates(SEED_FILE)
    all_expanded = []

    for template in tqdm(seed_templates, desc="Expanding seeds"):
        variations = expand_template(template, EXPANSION_FACTOR)
        all_expanded.extend(variations)

    print(f"Total generated templates before filtering: {len(all_expanded)}")

    # Optional filtering
    all_expanded = filter_templates(all_expanded)
    print(f"Total templates after filtering: {len(all_expanded)}")

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_expanded, f, indent=2, ensure_ascii=False)
    print(f"Expanded templates saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
