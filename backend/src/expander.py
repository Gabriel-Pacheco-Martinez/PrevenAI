"""
This script expands a small set of seed JSON templates into a larger dataset (~1,000 prompts)
using LangChain and a language model. 
"""

# General
import re
import json
import random
from pathlib import Path
from typing import List, Dict

# LLM
from tqdm import tqdm # For progress bars
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ======
# LLM settings and seed
random.seed(42)
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.7,
    max_tokens=300
)

# ======
# JSON: load seed templates
def load_seed_templates(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======
# Expander: generate variations using LLM for each seed
def expand_template(template: Dict, num_variations: int = 50) -> List[Dict]:
    expanded = []
    
    message = [
        SystemMessage(content="You are a professional preventative health assistant."),
        HumanMessage(
            content=(
                f"Generate {num_variations} diverse, professional, and self-contained variations "
                f"of the following Q&A template. Keep the same topic and tone. "
                f"Do not give personalized advice. Keep answers concise.\n\n"
                f"TEMPLATE:\n"
                f"Prompt: {template['prompt']}\n"
                f"Response: {template['response']}\n"
                f"Topic: {template['topic']}\n\n"
                f"Output format: JSON array with keys 'prompt', 'response', 'topic'."
            )
        ),
    ]

    # Call LLM to generate expanded templates
    response = llm.invoke(message)
    raw_text = response.content
    print("\n--- RAW RESPONSE ---")
    print(raw_text)

    # Clean common LLM wrappers like ```json ... ```
    cleaned_text = re.sub(r"^```json|```$", "", raw_text).strip()
    print("\n--- CLEANED RESPONSE ---")
    print(cleaned_text)
    
    # Parse output (expecting JSON array)
    try:
        variations = json.loads(cleaned_text)
        # Ensure topic is consistent
        for v in variations:
            v['topic'] = template['topic']
        expanded.extend(variations)
        print(f"[✓] Successfully expanded template: '{template['prompt']}' ({len(variations)} variations)")
    except Exception as e:
        print(f"[x] Error parsing JSON for template '{template['prompt']}': {e}")
    
    return expanded

# ======
# Filtering: remove duplicates or longer responses
def filter_templates(templates: List[Dict]) -> List[Dict]:
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
    # Files
    seed_templates = load_seed_templates("data/seed_templates.json")
    output_file = Path("data/expanded_templates.json")
    expansion_factor = 50

    # ====
    # Expand templates
    all_expanded = []
    for template in tqdm(seed_templates, desc="Expanding seeds"):
        variations = expand_template(template, expansion_factor)
        all_expanded.extend(variations)

    print(f"Total generated templates before filtering: {len(all_expanded)}")

    # ====
    # Filter for generated templates
    all_expanded = filter_templates(all_expanded)
    print(f"Total templates after filtering: {len(all_expanded)}")

    # ====
    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_expanded, f, indent=2, ensure_ascii=False)
    print(f"Expanded templates saved to {output_file}")


if __name__ == "__main__":
    main()
