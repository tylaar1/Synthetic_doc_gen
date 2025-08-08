import random
import json
import itertools
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime

CHARGES = {
    "security_fee": {
        "description": "A fee charged for security measures at the airport.",
        "variables": ["aircraft_weight", "passenger_count", "baggage_weight"],
        "synonyms": ["security charge", "safety fee"],
    },
    "landing_fee": {
        "description": "A fee charged for landing the aircraft at the airport.",
        "variables": ["aircraft_weight", "passenger_count", "fuel_consumption", "baggage_weight", "aircraft_type", "flight_type"],
        "synonyms": ["landing charge", "arrival fee"],
    },
    "fuel_tax": {
        "description": "A tax applied to the fuel consumed by the aircraft.",
        "variables": ["fuel_consumption", "aircraft_weight", "flight_type"],
        "synonyms": ["fuel charge", "fuel levy"],
    },
    "baggage_fee": {
        "description": "A fee charged for the baggage carried by passengers.",
        "variables": ["baggage_weight", "aircraft_type"],
        "synonyms": ["luggage charge", "baggage levy"],
    },
    "passenger_service_fee": {
        "description": "A fee charged for services provided to passengers at the airport.",
        "variables": ["passenger_count", "aircraft_type", "flight_type"],
        "synonyms": ["passenger charge", "service fee"],
    },
    "airport_facility_fee": {
        "description": "A fee charged for the use of airport facilities.",
        "variables": ["aircraft_type", "flight_type"],
        "synonyms": ["facility charge", "airport usage fee"],
    }
}

VARIABLES = {
    "aircraft_weight": {
        "dtype": float,
        "values": (5000.0, 400000.0),
        "units": "kg",
        "alternative_names": ["aircraft mass", "plane weight"]
    },
    "passenger_count": {
        "dtype": int,
        "values": (10, 500),
        "units": "passengers",
        "alternative_names": ["passenger number", "traveler count"]
    },
    "fuel_consumption": {
        "dtype": float,
        "values": (500.0, 5000.0),
        "units": "liters",
        "alternative_names": ["fuel usage", "fuel burn"]
    },
    "baggage_weight": {
        "dtype": float,
        "values": (5.0, 50.0),
        "units": "kg",
        "alternative_names": ["luggage weight", "baggage mass"]
    },
    "aircraft_type": {
        "dtype": str,
        "values": ["passenger", "cargo", "private", "military"],
        "units": "aircraft type",
        "alternative_names": ["plane type", "aircraft category"]
    },
    "flight_type": {
        "dtype": str,
        "values": ["domestic", "international", "charter"],
        "units": "flight type",
        "alternative_names": ["flight category", "flight class"]
    },
}

def create_categorical_conditions(var_name: str, values: List[str]) -> Tuple[str, str]:
    shuffled_values = random.sample(values, len(values))
    body_lines = []
    description_clauses = []
    for idx, val in enumerate(shuffled_values):
        line = f"{'if' if idx == 0 else 'elif'} {var_name} == '{val}': return {idx}"
        description = f"if {var_name} is '{val}', charge is {idx}"
        body_lines.append(f"    {line}")
        description_clauses.append(description)
    body_lines.append(f"    else: raise ValueError(f'Unknown value for {var_name}: {{{var_name}}}')")
    function_code = f"def compute_charge({var_name}):\n" + "\n".join(body_lines)
    full_description = "; ".join(description_clauses)
    return function_code, full_description

def create_numeric_conditions(
    var_name: str,
    val_range: Tuple[float, float],
    value_type: type = float
) -> Tuple[str, str]:
    min_val, max_val = val_range
    num_splits = random.randint(1, 3)
    if value_type == float:
        thresholds = sorted(round(random.uniform(min_val, max_val), 1) for _ in range(num_splits))
        fmt = lambda x: f"{round(x, 1):.1f}"
    elif value_type == int:
        thresholds = sorted(random.randint(int(min_val), int(max_val)) for _ in range(num_splits))
        fmt = lambda x: str(int(x))
    else:
        raise ValueError("value_type must be type float or int")
    split_bounds = [min_val] + thresholds + [max_val]
    if random.choice([True, False]):
        split_bounds = split_bounds[::-1]
    body_lines = []
    description_clauses = []
    for idx in range(len(split_bounds) - 1):
        lower = min(split_bounds[idx], split_bounds[idx+1])
        upper = max(split_bounds[idx], split_bounds[idx+1])
        lower_str = fmt(lower)
        upper_str = fmt(upper)
        line = f"{'if' if idx == 0 else 'elif'} {lower_str} <= {var_name} < {upper_str}: return {idx} * {var_name}"
        body_lines.append(f"    {line}")
        description_clauses.append(f"if {var_name} in [{lower_str}, {upper_str}), charge is {idx} × {var_name}")
    body_lines.append(f"    else: raise ValueError(f'{var_name} is out of expected range')")
    function_code = f"def compute_charge({var_name}):\n" + "\n".join(body_lines)
    full_description = "; ".join(description_clauses)
    return function_code, full_description

def create_two_variable_conditions(
    charge_name: str,
    cat_var: str,
    cont_var: str,
    cat_vals: List[str],
    cont_vals: Tuple[float, float]
) -> Tuple[str, str]:
    code_lines: List[str] = []
    description_clauses = f"{charge_name} conditions by {cat_var} and {cont_var}:\n"
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
        intermediate_description = f"\nfor {val} the following charges apply:"
        multiplier = 1 + outer_idx if cont_dtype == int else 1 + (outer_idx / 10)
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
                f"  - if {cont_var} in [{lower_str}, {upper_str}), charge is {charge_str} × {cont_var}"
            )
            intermediate_code_lines.append(condition_line)
            intermediate_description += f"\n{description_line}"
        code_lines.extend(intermediate_code_lines)
        description_clauses += intermediate_description
    code_lines.append(f"else: raise ValueError(f'{cat_var} or {cont_var} is out of expected range')")
    body = "\n".join(code_lines)
    return body, description_clauses

def handle_variable(name: str) -> Tuple[str, str]:
    var = VARIABLES.get(name)
    if var is None:
        raise KeyError(f"Variable '{name}' not found.")
    dtype = var["dtype"]
    values = var["values"]
    units = var["units"]
    if dtype == str:
        code, description = create_categorical_conditions(name, values)
        # omit units sentence for categoricals to avoid awkward phrasing
    elif dtype in (int, float):
        code, description = create_numeric_conditions(name, values, dtype)
        if units:
            description += f" The variable '{name}' is measured in {units}."
    else:
        raise TypeError(f"Unsupported dtype for variable '{name}': {dtype}")
    return description, code

def handle_two_variable_condition(charge_name: str, cat_var: str, cont_var: str) -> Tuple[str, str, List[str]]:
    cat_vals = VARIABLES[cat_var]["values"]
    cont_vals = VARIABLES[cont_var]["values"]
    cat_dtype = VARIABLES[cat_var]["dtype"]
    cont_dtype = VARIABLES[cont_var]["dtype"]
    if cat_dtype != str or cont_dtype not in (float, int):
        raise ValueError(f"Unsupported variable types: {cat_var} ({cat_dtype}), {cont_var} ({cont_dtype})")
    body, description = create_two_variable_conditions(charge_name, cat_var, cont_var, cat_vals, cont_vals)
    # wrap into a full function for consistency with single-variable path
    code = f"def compute_charge({cat_var}, {cont_var}):\n    " + body.replace("\n", "\n    ")
    return description, code, [cat_var, cont_var]

def modify_description(description: str, var_name: Union[str, List[str], Tuple[str, ...]]) -> str:
    if isinstance(var_name, str):
        alt_name = random.choice(VARIABLES[var_name]["alternative_names"])
        return description.replace(var_name, alt_name)
    else:
        for var in list(var_name):
            alt_name = random.choice(VARIABLES[var]["alternative_names"])
            description = description.replace(var, alt_name)
        return description

def select_and_handle_variables(charge_name: str) -> Tuple[str, str, List[str]]:
    if charge_name not in CHARGES:
        raise ValueError(f"Charge '{charge_name}' not found in CHARGES.")
    variables = CHARGES[charge_name]["variables"]
    if not variables:
        raise ValueError(f"No variables available for charge '{charge_name}'.")
    var_pairs = list(itertools.combinations(variables, 2))
    valid_pairs = []
    for v1, v2 in var_pairs:
        dtype1 = VARIABLES[v1]["dtype"]
        dtype2 = VARIABLES[v2]["dtype"]
        if (dtype1 == str and dtype2 in (int, float)) or (dtype2 == str and dtype1 in (int, float)):
            valid_pairs.append((v1, v2))
    if valid_pairs and random.random() < 0.5:
        cat_var, cont_var = random.choice(valid_pairs)
        if VARIABLES[cat_var]["dtype"] != str:
            cat_var, cont_var = cont_var, cat_var
        desc, code, used = handle_two_variable_condition(charge_name, cat_var, cont_var)
        return desc, code, used
    var = random.choice(variables)
    desc, code = handle_variable(var)
    return desc, code, [var]

def generate_output_structure() -> Dict[str, Any]:
    output_structure: Dict[str, Any] = {}
    for charge_name, charge_info in CHARGES.items():
        desc, code, used_vars = select_and_handle_variables(charge_name)
        desc = modify_description(desc, used_vars)
        output_structure[charge_name] = {
            "description": desc,
            "code": code,
            "variables_used": used_vars,
            "synonyms": charge_info.get("synonyms", []),
            "charge_description": charge_info["description"],
        }
    return output_structure

if __name__ == "__main__":
    output_structure = generate_output_structure()

    # Generate an identifier like 2025-08-08_11-42-05
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_structure["run_id"] = run_id

    with open(f"output_structure_{run_id}.json", "w") as f:
        json.dump(output_structure, f, indent=2)

