import json
import random
import re
from pathlib import Path
from typing import Any, Optional
import pandas as pd


def generate_schema(output_path: str, data_path: Optional[str]=None, column_info: Optional[dict]=None):
    """Generate the schema information for the given data file.

        Args:
            output_path (str): The path to save the schema information.
            data_path (Optional[str], optional): The path to the data file. Defaults to None.
            column_info (Optional[dict], optional): A dictionary containing column information for a single table.
                The keys are column names and the values are column types. Defaults to None.

        Returns:
            schema_list (List[dict]): A list of dictionaries containing the schema information.
            output_path (str): The path to the schema information.
        """

    # Load the data file
    df = pd.read_csv(data_path) if data_path else None
    # Extract the schema information
    # column_info is a dictionary for a single table with column names as keys and column types as values
    # TODO Extend column_info to support multiple tables.
    schema = df.dtypes.to_dict() if df is not None else column_info
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
            unique_values = df[key].dropna().unique().tolist() if df is not None else []
            if len(unique_values) > 10:
                # Randomly sample 10 values
                unique_values = random.sample(unique_values, k=10)
            if not unique_values or (len(unique_values) > 0 and not bool(re.search(r"[A-Za-z]", unique_values[0]))):
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
