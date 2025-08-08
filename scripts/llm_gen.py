import json
import re
import random
import argparse
import glob
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompts import one_shot_example


class LLMClient:
    """Simple client for interacting with Hugging Face Transformers models."""
    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct', device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )

    def generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        system_prompt = (
            "You are a professional assistant trained to generate policy documentation and HTML tables for airport pricing documents. "
            "You always follow instructions precisely and do not include any explanation, commentary, or additional text beyond what is requested. "
            "When asked to generate a table, output only valid HTML starting with a <table> tag. "
            "When asked to write a paragraph, respond in a concise, informative, and professional tone. "
            "Do not refer to the user, and do not include disclaimers or formatting notes. "
            "Maintain consistency with real-world pricing policies and use appropriate domain terminology."
            "The goal is to create documentation as close to human-written as possible, Do not use any symbols or formatting a human would not use. "
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.generator(
            chat_prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        return generated[len(chat_prompt):].strip()


def extract_table(html_str: str) -> str:
    m = re.search(r"<table.*?>.*?</table>", html_str, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else "This charge is temporarily suspended due to technical issues."


def generate_document_table(llm_client: LLMClient, airport_name: str, charge_description: str, max_retries: int = 3) -> str:
    prompt = f"""Write a professionally human readable formatted html table for a charging policy document from {airport_name}. 
You should only return the HTML table code, nothing else.
All information should be contained within a single table not multiple tables.
An example will be provided to help you understand the format however you should add your own stylistic touch.
The example is:
{one_shot_example} 
INPUT: {charge_description}.  
OUTPUT:"""
    max_tokens = 1000
    for attempt in range(max_retries):
        html = llm_client.generate_text(prompt, max_tokens=max_tokens)
        if "<table" in html.lower() and "</table>" in html.lower():
            return html
        if attempt == 1:
            max_tokens = 2000
    return ""


def generate_charge_description(llm_client: LLMClient, airport_name: str, charge_description: str) -> str:
    prompt = f"""Write a professional description for a charge category at {airport_name}. 
The description should be concise, informative, and suitable for a charging policy document. 
The charge description is: {charge_description}. 
The description should be 1-2 paragraphs explaining what the charge covers and its purpose.
Description should be clear and suitable for a formal document.
Do not include any disclaimers or additional text, just the description. Do not hallucinate any charges not in the description."""
    return llm_client.generate_text(prompt, max_tokens=500)


def decide_charge_format(llm_client: LLMClient, airport_name: str, entry: Dict[str, Any], table_chance: float = 0.7) -> str:
    desc = entry["description"]
    used = entry.get("variables_used", [])
    want_table = len(used) > 1 or random.random() < table_chance
    if want_table:
        html = generate_document_table(llm_client, airport_name, desc, max_retries=2)
        return extract_table(html) if html else generate_charge_description(llm_client, airport_name, desc)
    return generate_charge_description(llm_client, airport_name, desc)


def generate_airport_name(llm_client: LLMClient, currency: str) -> str:
    examples = {
        "USD": "Liberty International Airport, Eagle Point Aviation Hub, Sunrise Municipal Airport, Golden Gate Regional Airport",
        "EUR": "Europa International Airport, Alpine Valley Airport, Mediterranean Coast Airport, Northern Lights Aviation Hub",
        "GBP": "Royal Crown Airport, Thames Valley International, Highland Regional Airport, Coastal Wings Airport",
    }
    prompt = f"""Create a fictional airport name for a region that used {currency}. 
Use these examples as inspiration: {examples.get(currency, examples['GBP'])}
Generate one creative, professional-sounding airport name that fits the regional style. 
Only return the airport name, nothing else."""
    return llm_client.generate_text(prompt, max_tokens=50)


def make_section_title(llm_client: LLMClient, airport_name: str, category: str) -> str:
    prompt = f"""Create a professional section title for a charging policy document from {airport_name}. 
The title should be very concise and clearly indicate the charge category(s). 
The category is: {category}. 
Keep it formal and suitable for an official document, no more than 20 words."""
    return llm_client.generate_text(prompt, max_tokens=20)


def generate_document_introduction(llm_client: LLMClient, airport_name: str, currency: str) -> str:
    prompt = f"""Write a professional introduction for a charging policy document from {airport_name}. 
This document outlines the airport's fee structure and charging policies for aviation services in {currency}. 
Keep it formal but welcoming, about 2-3 sentences. Include the airport name and mention this is an official policy document."""
    return llm_client.generate_text(prompt, max_tokens=150)


def generate_document_conclusion(llm_client: LLMClient, airport_name: str, currency: str) -> str:
    prompt = f"""Write a professional conclusion for a charging policy document from {airport_name}. 
The policies cover various aviation services charged in {currency}. 
Include contact information for questions. Keep it formal and brief, 2-3 sentences."""
    return llm_client.generate_text(prompt, max_tokens=150)


def extract_first_paragraph(text: str) -> str:
    m = re.search(r'(.*?)(?:\n\s*\n|$)', text.strip(), re.DOTALL)
    return m.group(1).strip() if m else ""


def choose_currency() -> str:
    return random.choice(["USD", "EUR", "GBP"])


def load_generated_charges(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, dict) and "description" in v and "code" in v}


def find_latest_output_json(pattern: str = "output_structure_*.json") -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found.")
    # Prefer by timestamp in filename if possible, fallback to file mtime
    try:
        files_sorted = sorted(files, key=lambda f: datetime.strptime(os.path.basename(f)[17:-5], "%Y-%m-%d_%H-%M-%S"), reverse=True)
    except Exception:
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    return files_sorted[0]


def generate_document_from_json(client: LLMClient, output_json_path: str) -> Tuple[str, List[str], List[str]]:
    data = load_generated_charges(output_json_path)
    currency = choose_currency()
    airport_name = extract_first_paragraph(generate_airport_name(client, currency))
    introduction = generate_document_introduction(client, airport_name, currency)
    conclusion = generate_document_conclusion(client, airport_name, currency)

    charge_names = list(data.keys())
    titles = [make_section_title(client, airport_name, charge.replace("_", " ").title()) for charge in charge_names]
    rendered_sections, answers, names = [], [], []

    for charge, title in zip(charge_names, titles):
        entry = data[charge]
        section_html_or_text = decide_charge_format(client, airport_name, entry)
        rendered_sections.append(f"## {title}\n\n{section_html_or_text}")
        answers.append(entry.get("code", ""))
        names.append(charge)

    doc = f"# {airport_name} Charging Policy Document\n\n{introduction}\n\n"
    doc += "\n\n".join(rendered_sections)
    doc += f"\n\n{conclusion}\n\n"
    doc += f"**Policy Effective Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
    return doc, answers, names


def create_charge_answer_pair(answers: List[str], variables: List[str]) -> Tuple[str, str]:
    if not answers or not variables:
        return "", ""
    idx = random.randint(0, len(answers) - 1)
    return answers[idx], variables[idx]


def save_document(content: str, filename: str = None) -> str:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_dataset/charging_policy_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def save_datapoint_to_jsonl(datapoint: Dict[str, Any], filename: str = "synthetic_dataset/datapoints.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(datapoint, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Path to a specific output_structure JSON file", default=None)
    args = parser.parse_args()

    if args.json:
        json_path = args.json
    else:
        json_path = find_latest_output_json()

    client = LLMClient(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
    document_content, answers, names = generate_document_from_json(client, json_path)

    target, charge_type = create_charge_answer_pair(answers, names)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    datapoint = {
        "charge_type": charge_type,
        "document": document_content,
        "target": target,
        "document_id": run_id,
    }
    save_datapoint_to_jsonl(datapoint)

    saved_filename = save_document(document_content, filename=f"synthetic_dataset/document_{run_id}.md")
    print(f"Using JSON file: {json_path}")
    print(f"Document saved to {saved_filename}")
