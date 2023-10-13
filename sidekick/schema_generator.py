import json
import re
import pandas as pd


def generate_schema(data_path, output_path):
    df = pd.read_csv(data_path)
    # Extract the schema information
    schema = df.dtypes.to_dict()
    schema_list = []
    special_characters = {" ": "_", ":": "_", "/": "_", "-": "_"}
    syntax_names = ["default"]

    for key, value in schema.items():
        new_key = "".join(special_characters[s] if s in special_characters.keys() else s for s in key)
        if new_key.lower() in syntax_names:
            new_key = new_key + "_col"
        if value == "object":
            value = "TEXT"
            unique_values = df[key].dropna().unique().tolist()
            if not bool(re.search(r"[A-Za-z]", unique_values[0])):
                schema_list.append({"Column Name": new_key, "Column Type": value})
            else:
                schema_list.append({"Column Name": new_key, "Column Type": value, "Sample Values": unique_values})
        else:
            value = "NUMERIC"
            schema_list.append({"Column Name": new_key, "Column Type": value})

    # Save the schema information to a JSONL format
    with open(output_path, "w") as f:
        for item in schema_list:
            json.dump(item, f)
            f.write("\n")
    return output_path
