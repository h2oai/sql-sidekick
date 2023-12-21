import json
import random
import re
from pathlib import Path

import pandas as pd


def generate_schema(data_path: str, output_path: str):
    """ Generate the schema information for the given data file
    Args:
        data_path: The path to the data file
        output_path: The path to save the schema information
    Returns:
        schema_list: A list of dictionaries containing the schema information
        output_path: The path to the schema information
    """

    # Load the data file
    df = pd.read_csv(data_path)
    # Extract the schema information
    schema = df.dtypes.to_dict()
    schema_list = []
    special_characters = {" ": "_", ":": "_", "/": "_", "-": "_", "(": "", ")": "", ".": "_"}
    syntax_names = ["default"]

    for key, value in schema.items():
        new_key = "".join(special_characters[s] if s in special_characters.keys() else s for s in key)
        if new_key.lower() in syntax_names:
            new_key = new_key + "_col"
        if new_key[0].isdigit():
            _temp = "".join((new_key[:0], "Digit_", new_key[1:]))
            new_key = _temp
        if value == "object":
            value = "TEXT"
            unique_values = df[key].dropna().unique().tolist()
            if len(unique_values) > 10:
                # Randomly sample 10 values
                unique_values = random.sample(unique_values, k=10)
            if not bool(re.search(r"[A-Za-z]", unique_values[0])):
                schema_list.append({"Column Name": new_key, "Column Type": value})
            else:
                schema_list.append({"Column Name": new_key, "Column Type": value, "Sample Values": unique_values})
        else:
            value = "NUMERIC"
            schema_list.append({"Column Name": new_key, "Column Type": value})

    # Save the schema information to a JSONL format
    if not Path(output_path).exists():
        f = Path(output_path)
        f.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in schema_list:
            json.dump(item, f)
            f.write("\n")
    return schema_list, output_path
