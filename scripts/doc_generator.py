import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import requests
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
from prompts import one_shot_example


VARIABLES = { # next step - add alternative names to pass to the LLM
    "aircraft_weight": {
        "dtype": float,
        "values": (5000.0, 400000.0),  # in kilograms
        "units": "kg",
        "alternative_names": ["aircraft mass", "plane weight"]
    },
    "passenger_count": {
        "dtype": int,
        "values": (10, 500),  # Number of passengers
        "units": "passengers",
        "alternative_names": ["passenger number", "traveler count"]
    },
    "landing_fee_rate": {
        "dtype": float,
        "values": (20.0, 100.0),  # per tonne, rate for landing fees
        "units": "tonnes",
        "alternative_names": ["landing charge", "landing cost"]
    },
    "fuel_consumption": {
        "dtype": float,
        "values": (500.0, 5000.0),  # in liters
        "units": "liters",
        "alternative_names": ["fuel usage", "fuel burn"]
    },
    "baggage_weight": {
        "dtype": float,
        "values": (5.0, 50.0),  # in kilograms, baggage weight
        "units": "kg",
        "alternative_names": ["luggage weight", "baggage mass"]
    },
    "airport_security_rate": {
        "dtype": float,
        "values": (1.0, 10.0),  # per passenger, security levy
        "units": "passengers",
        "alternative_names": ["security charge", "security fee"]
    },
    "hanger_fee_rate": {
        "dtype": int,
        "values": (1, 31),  # per day, hangar fee rate
        "units": "days",
        "alternative_names": ["hangar charge", "hangar cost"]
    },
    "parking_fee_rate": {
        "dtype": float,
        "values": (0.0, 20.0),  # per hour, aircraft parking rate
        "units": "hours",
        "alternative_names": ["parking charge", "parking cost"]
    },
    "aircraft_type": {
        "dtype": str,
        "values": ["passenger", "cargo", "private", "military"],  # Type of aircraft operating
        "units": "aircraft type",
        "alternative_names": ["plane type", "aircraft category"]
    },
    "flight_type": {
        "dtype": str,
        "values": ["domestic", "international", "charter"],  # Type of flight
        "units": "flight type",
        "alternative_names": ["flight category", "flight class"]
    },
}

def select_variables(n_vars: int, var_names: List[str]) -> List[str]:
    """Select a random subset of variable names."""
    return random.sample(var_names, n_vars)

def handle_variable(name: str):
    var = VARIABLES.get(name)
    if var is None:
        raise KeyError(f"Variable '{name}' not found.")
    
    dtype = var["dtype"]
    values = var["values"]
    units = var["units"]
    
    if dtype == str:
        answer,description,_=create_categorical_conditions(name, values) # will return properly when generating x,y pairs not just x
        description += f" The charge is measured with the following units ({units})" if units else ""
    elif dtype in (int, float):
        answer, description, _ = create_numeric_conditions(name, values, dtype)
        description += f" The charge is measured with the following units ({units})" if units else ""
    else:
        print(f"Unsupported dtype for variable '{name}': {dtype}")
        description = None
    return description , answer


def create_categorical_conditions(var_name: str, values: List[str]) -> Tuple[List[str], str, Dict[str, int]]:
    """Generate executable conditions, a single description string, and a charge lookup for a categorical variable."""
    shuffled_values = random.sample(values, len(values))
    
    code_lines = []
    description_clauses = []
    charges = {}

    for idx, val in enumerate(shuffled_values):
        condition = f"{'if' if idx == 0 else 'elif'} {var_name} == '{val}': return {idx}"
        description = f"if {var_name} is '{val}', charge is {idx}"

        code_lines.append(condition)
        description_clauses.append(description)
        charges[val] = idx

    code_lines.append(f"else: raise ValueError(f'Unknown value for {var_name}: {{ {var_name} }}')")

    full_description = "; ".join(description_clauses)

    return code_lines, full_description, charges
 #for now return code lines and charges as unsure which will be most useful

def create_numeric_conditions(
    var_name: str,
    val_range: Tuple[float, float],
    value_type: str = "float"
) -> Tuple[List[str], str, List[Dict[str, float]]]:
    """Generate if-elif return conditions and description for a numeric variable (int or float), with consistent formatting."""
    
    min_val, max_val = val_range
    num_splits = random.randint(1, 3)

    if value_type == float:
        thresholds = sorted(round(random.uniform(min_val, max_val), 1) for _ in range(num_splits))
        fmt = lambda x: f"{round(x, 1):.1f}"
    elif value_type == int:
        thresholds = sorted(random.randint(int(min_val), int(max_val)) for _ in range(num_splits))
        fmt = lambda x: f"{int(x)}"
    else:
        raise ValueError("value_type must be 'float' or 'int'")

    split_bounds = [min_val] + thresholds + [max_val]
    if random.choice([True, False]):
        split_bounds = split_bounds[::-1]

    code_lines = []
    description_clauses = []
    charge_bands = []

    for idx in range(len(split_bounds) - 1):
        lower = min(split_bounds[idx], split_bounds[idx+1])
        upper = max(split_bounds[idx], split_bounds[idx+1])

        lower_str = fmt(lower)
        upper_str = fmt(upper)

        condition_line = f"{'if' if idx == 0 else 'elif'} {lower_str} <= {var_name} < {upper_str}: return {idx} * {var_name}"
        description_clause = f"if {var_name} in [{lower_str}, {upper_str}), charge is {idx} × {var_name}"
        
        code_lines.append(condition_line)
        description_clauses.append(description_clause)
        charge_bands.append({
            "min": float(fmt(lower)) if value_type == float else int(fmt(lower)),
            "max": float(fmt(upper)) if value_type == float else int(fmt(upper)),
            "multiplier": idx
                })

    code_lines.append(f"else: raise ValueError(f'{var_name} is out of expected range')")
    full_description = "; ".join(description_clauses)

    return code_lines, full_description, charge_bands


#lots of overlaping code here should likely be turned into a single function in future
def create_two_variable_conditions(cat_var: str,cont_var: str,cat_vals: List[str],cont_vals: Tuple[float, float]) -> Tuple[List[str], str]:
    """
    Combine a categorical variable with a numeric continuous variable (int or float)
    to create a grid of charge conditions. Outputs code lines and natural language description.
    """

    code_lines = []
    description_clauses = f"charge conditions for {cont_var} and {cat_var}:\n"
    
    # Detect type of cont_var from VARIABLES dict
    cont_dtype = VARIABLES[cont_var]["dtype"]
    min_val, max_val = cont_vals
    num_splits = random.randint(1, 3)

    if cont_dtype == float:
        thresholds = sorted(round(random.uniform(min_val, max_val), 1) for _ in range(num_splits))
        fmt = lambda x: f"{round(x, 1):.1f}"
    elif cont_dtype == int:
        thresholds = sorted(random.randint(int(min_val), int(max_val)) for _ in range(num_splits))
        fmt = lambda x: f"{int(x)}"
    else:
        raise ValueError(f"Unsupported continuous dtype: {cont_dtype}")

    split_bounds = [min_val] + thresholds + [max_val]

    if random.choice([True, False]):
        split_bounds = split_bounds[::-1]

    for outer_idx, val in enumerate(cat_vals):
        intermediate_code_lines = []
        intermediate_description_clauses = f"\nfor {val} the following charges apply:"

        # Use float or int multiplier logic depending on type
        if cont_dtype == int:
            multiplier = 1 + outer_idx  # e.g. 1, 2, 3
        else:
            multiplier = 1 + (outer_idx / 10)  # e.g. 1.0, 1.1, 1.2

        for idx in range(len(split_bounds) - 1):
            lower = min(split_bounds[idx], split_bounds[idx+1])
            upper = max(split_bounds[idx], split_bounds[idx+1])

            lower_str = fmt(lower)
            upper_str = fmt(upper)
            charge_str = round(idx * multiplier, 2) if cont_dtype == float else int(idx * multiplier)

            condition_line = (
                f"{'if' if idx == 0 and outer_idx == 0 else 'elif'} "
                f"{cat_var} == '{val}' and {lower_str} <= {cont_var} < {upper_str}: "
                f"return {idx} * {cont_var} * {multiplier}"
            )
            description_line = (
                f"  - if {cont_var} in [{lower_str}, {upper_str}), "
                f"charge is {charge_str} × {cont_var}"
            )

            intermediate_code_lines.append(condition_line)
            intermediate_description_clauses += f"\n{description_line}"

        code_lines.extend(intermediate_code_lines)
        description_clauses += intermediate_description_clauses

    code_lines.append(f"else: raise ValueError(f'{cat_var} or {cont_var} is out of expected range')")
    return code_lines, description_clauses


def handle_two_variable_condition(cat_var: str, cont_var: str) -> Tuple[str, str]:
    """Handle two variable conditions by creating a description and code lines."""
    cat_vals = VARIABLES[cat_var]["values"]
    cont_vals = VARIABLES[cont_var]["values"]
    cat_dtype = VARIABLES[cat_var]["dtype"]
    cont_dtype = VARIABLES[cont_var]["dtype"]
    if cat_dtype != str or cont_dtype not in (float, int): # should expand to allow int as well
        raise ValueError(f"Unsupported variable types: {cat_var} ({cat_dtype}), {cont_var} ({cont_dtype})")
    
    code_lines, description = create_two_variable_conditions(cat_var, cont_var, cat_vals, cont_vals)
    
    return description, "\n".join(code_lines)

def pair_variables(variables: List[str], n_pairs: int) -> Tuple[List[List[str]], List[str]]:
    '''Sort variables into pairs of categorical and continuous variables.'''
    # first separate categorical and continuous variables
    categorical_vars = [var for var in variables if VARIABLES[var]["dtype"] == str]
    float_vars = [var for var in variables if VARIABLES[var]["dtype"] == float]
    integer_vars = [var for var in variables if VARIABLES[var]["dtype"] == int]
    pairs = []
    # then create pairs of categorical and continuous variables
    for _ in range(n_pairs):
        if not categorical_vars or not (float_vars or integer_vars):
            break
        cat_var = random.choice(categorical_vars)
        cont_var = random.choice(float_vars + integer_vars)
        pairs.append([cat_var, cont_var])
        categorical_vars.remove(cat_var) 
        if cont_var in float_vars:
            float_vars.remove(cont_var)
        else:
            integer_vars.remove(cont_var)
    remaining_vars = categorical_vars + float_vars + integer_vars
    return pairs, remaining_vars
        

def add_surcharge_conditions(var_name: str, code_lines: List[str], var_dtype: type):
    """if variable is a float [or int], add surcharge conditions to the code and generate a description."""
    pass

def modify_description(description: str, var_name: Union[str, List[str]]) -> str:
    """Replace occurrences of one or more variable names in description with random alternatives."""
    if isinstance(var_name, str):
        alt_name = random.choice(VARIABLES[var_name]["alternative_names"])
        return description.replace(var_name, alt_name)
    elif isinstance(var_name, list):
        for var in var_name:
            alt_name = random.choice(VARIABLES[var]["alternative_names"])
            description = description.replace(var, alt_name)
        return description

    else:
        raise ValueError("var_name must be a string or list of strings.")

class LLMClient:
    """Simple client for interacting with Hugging Face Transformers models."""

    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct', device: str = None):
        """
        model_name: the name or path of the HF model
        device: 'cuda', 'cpu', or 'mps'; if None, auto-detect
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device == "cuda" else -1)

    def generate_text(self, prompt: str, max_tokens: int = 200) -> str:
        messages = []
        system_prompt = (
    "You are a professional assistant trained to generate policy documentation and HTML tables for airport pricing documents. "
    "You always follow instructions precisely and do not include any explanation, commentary, or additional text beyond what is requested. "
    "When asked to generate a table, output only valid HTML starting with a <table> tag. "
    "When asked to write a paragraph, respond in a concise, informative, and professional tone. "
    "Do not refer to the user, and do not include disclaimers or formatting notes. "
    "Maintain consistency with real-world pricing policies and use appropriate domain terminology."
    "The goal is to create documentation as close to human-written as possible, Do not use any symbols or formatting a human would not use. "
    )
        messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})
        
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
  
    
def generate_document_table(llm_client: LLMClient,airport_name: str,charge_description: str,max_retries: int = 3) -> str:
    """Generate a table for the charging policy document with retry logic."""
    prompt = f"""Write a professionally human readable formatted html table for a charging policy document from {airport_name}. 
    You should only return the HTML table code, nothing else.
    All information should be contained within a single table not multiple tables.
    An example will pe provided to help you understand the format however you should add your own stylistic touch.
    The example is:
    {one_shot_example} 
    INPUT: {charge_description}.  
    OUTPUT: """
    # Note gpt 4 able to do this without example so if wanting to produce more varied outputs using gpt 4 remove example.
    
    max_tokens = 1000  # Initial max tokens for the first attempt
    for attempt in range(0, max_retries):
        
        html = llm_client.generate_text(prompt, max_tokens=max_tokens)
        # Check if the output contains a valid <table> tag
        if "<table" in html.lower() and "</table>" in html.lower():
            return html 
        # Increase max_tokens after first attempt
        if attempt == 1:
            max_tokens = 2000

    # Soft fail
    print(f"Failed to generate valid table for {charge_description} after {max_retries} attempts. Returning empty string.", flush=True)
    return ""

# function redundant until logic to create surcharge conditions is added
def add_surcharge_to_table(llm_client: LLMClient, table_html: str, surcharge_description: str) -> str:
    """Add a surcharge row to an existing HTML table."""
    prompt = f"""You are a professional assistant trained to generate policy documentation and HTML tables for airport pricing documents. 
    You always follow instructions precisely and do not include any explanation, commentary, or additional text beyond what is requested.
    Add a new row to the following HTML table with the description: {surcharge_description}. 
    The table is:
    {table_html} """
    return(llm_client.generate_text(prompt, max_tokens=1500))


def extract_table(html_str: str) -> str:
    match = re.search(r"<table.*?>.*?</table>", html_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return "This charge is temporarily suspended due to technical issues." # will need to be properly handled when generating x,y pairs not just x

def generate_charge_description(llm_client: LLMClient,airport_name: str,charge_description: str) -> str:
    'generates a professional sounding description for a charge category'
    prompt = f"""Write a professional description for a charge category at {airport_name}. 
    The description should be concise, informative, and suitable for a charging policy document. 
    The charge description is: {charge_description}. 
    The description should be 1-2 paragraphs explaining what the charge covers and its purpose.
    description should be clear and suitable for a formal document.
    The description should cover the charge in its entirety, including any relevant details about the charge.
    Do not include any disclaimers or additional text, just the description. Do not hallucinate any charges not in the description."""
    return llm_client.generate_text(prompt, max_tokens=500)

def decide_charge_format(llm_client: LLMClient, airport_name: str, desc: str,charge_name, table_chance: float = 0.7) -> str:
    """if charge is complex enough, generate a table, otherwise randomly choose between a table and a description."""
    if len(charge_name) > 1 or random.random() < table_chance:  # 70% chance to generate a table
        table = generate_document_table(llm_client, airport_name, desc,2)
        return extract_table(table) 
    else:  
        return generate_charge_description(llm_client, airport_name, desc) 

def generate_airport_name(llm_client: LLMClient, currency: str) -> str:
    examples = {
        "USD": "Liberty International Airport, Eagle Point Aviation Hub, Sunrise Municipal Airport, Golden Gate Regional Airport",
        "EUR": "Europa International Airport, Alpine Valley Airport, Mediterranean Coast Airport, Northern Lights Aviation Hub",
        "GBP": "Royal Crown Airport, Thames Valley International, Highland Regional Airport, Coastal Wings Airport",
    }

    prompt = f"""Create a fictional airport name for a region that used {currency}. 
    Use these examples as inspiration: {examples.get(currency, examples[currency])}
    Generate one creative, professional-sounding airport name that fits the regional style. 
    Only return the airport name, nothing else."""

    return llm_client.generate_text(prompt, max_tokens=50)

def make_section_title(llm_client: LLMClient, airport_name: str, category: str) -> str:
    """Generate a section title for a charge category."""
    prompt = f"""Create a professional section title for a charging policy document from {airport_name}. 
    The title should be very concise and clearly indicate the charge category(s). 
    The category is: {category}. 
    Keep it formal and suitable for an official document, no more than 20 words."""
    
    return llm_client.generate_text(prompt, max_tokens=20)

def generate_document_introduction(llm_client: LLMClient, airport_name: str, currency: str) -> str:
    """Generate introduction for the charging policy document."""
    prompt = f"""Write a professional introduction for a charging policy document from {airport_name}. 
    This document outlines the airport's fee structure and charging policies for aviation services in {currency}. 
    Keep it formal but welcoming, about 2-3 sentences. Include the airport name and mention this is an official policy document."""

    return llm_client.generate_text(prompt, max_tokens=150)

def generate_category_preamble(llm_client: LLMClient, category: str, airport_name: str,premable_chance: float = 0.7) -> str:
    """Generate preamble for a charge category, or return empty string."""
    if random.random() > premable_chance:  # 30% chance of no preamble
        return ""

    prompt = f"""Write a brief professional paragraph explaining {category} charging policies at {airport_name}. 
    Keep it to 1-2 sentences, explaining what these charges cover and the pricing structure. 
    Be concise and informative about the policy, DO NOT mention specific charges. Give only this preamble, nothing else.
    Do not interact with the user, just return the preamble."""

    return llm_client.generate_text(prompt, max_tokens=100)

def generate_document_conclusion(llm_client: LLMClient, airport_name: str, currency: str) -> str:
    """Generate conclusion for the charging policy document."""
    prompt = f"""Write a professional conclusion for a charging policy document from {airport_name}. 
    The policies cover various aviation services charged in {currency}. 
    Include contact information for questions. Keep it formal and brief, 2-3 sentences."""

    return llm_client.generate_text(prompt, max_tokens=150)

def extract_first_paragraph(text: str) -> str:
    match = re.search(r'(.*?)(?:\n\s*\n|$)', text.strip(), re.DOTALL)
    return match.group(1).strip() if match else ""

def choose_currency() -> str:
    """Randomly select a currency from the predefined list."""
    currencies = ["USD", "EUR", "GBP"]
    return random.choice(currencies)

def generate_document(client: LLMClient) -> str:
    currency = choose_currency()
    airport_name_out = generate_airport_name(client, currency)
    airport_name = extract_first_paragraph(airport_name_out)

    introduction = generate_document_introduction(client, airport_name, currency)
    conclusion = generate_document_conclusion(client, airport_name, currency)

    variables = list(VARIABLES.keys())
    n_vars = 8  # Number of variables to select
    selected_variables = select_variables(n_vars, variables)
    n_pairs = 2  # Number of variable pairs to create
    paired,unpaired = pair_variables(selected_variables, n_pairs)
    names = paired + unpaired
    section_titles = [make_section_title(client, airport_name, name) for name in names]
    desc_ans_pairs = [handle_two_variable_condition(cat, cont) for cat, cont in paired]
    desc_ans_pairs.extend([handle_variable(var) for var in unpaired])
    descriptions = [pair[0] for pair in desc_ans_pairs]  # Extract descriptions
    modified_descriptions = [modify_description(desc, name) for desc, name in zip(descriptions, names)]
    answers = [pair[1] for pair in desc_ans_pairs]  # Extract answers
    premables = [generate_category_preamble(client, desc, airport_name) for desc in modified_descriptions]
    short_preambles = [extract_first_paragraph(p) for p in premables]
    generated = [decide_charge_format(client, airport_name, desc,name) for desc,name in zip(modified_descriptions,names)]

    sections_with_titles = [
        f"## {name}\n\n{preamble}\n\n{section}"
        for name, preamble, section in zip(section_titles, short_preambles, generated)
    ]

    document_content = f"# {airport_name} Charging Policy Document\n\n{introduction}\n\n"
    document_content += "\n\n".join(sections_with_titles)
    document_content += f"\n\n{conclusion}\n\n"
    document_content += f"**Policy Effective Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"

    return document_content, answers, names # could return section titles instead of names??

def create_charge_answer_pair(answers: List[str], variables: List[str]) -> Tuple[str,str]:
    """select a random answer and variable pair from the generated answers and variables."""
    if not answers or not variables:
        return "", ""
    idx = random.randint(0, len(answers) - 1)
    return answers[idx], variables[idx]


def save_document(content: str, filename: str = None) -> str:
    """Save document to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/gpfs01/home/ppytr13/rdc-analysis/documents/charging_policy_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


def save_datapoint_to_jsonl(datapoint, filename=f"synthetic_dataset/datapoints.jsonl"):
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(datapoint, f)
        f.write("\n")


if __name__ == "__main__":
    client = LLMClient(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
    document_content, answers, names = generate_document(client)
    target, charge_type = create_charge_answer_pair(answers, names)
    datapoint = {
    "charge_type": charge_type,
    "document": document_content,
    "target": target,
    "document_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    save_datapoint_to_jsonl(datapoint)


    saved_filename = save_document(document_content, filename=f"synthetic_dataset/document_{datapoint['document_id']}.md")
    print(f"Document saved to {saved_filename}")
    
