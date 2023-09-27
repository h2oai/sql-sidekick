import pandas as pd
import json
import re
import click

@click.command()
@click.option('--csv',
              default="data.csv",
              help='Enter the path of csv',
              type=str)

@click.option('--jsonl',
              default="table_info.jsonl",
              help='Enter the path of generated schema in jsonl',
              type=str)

def generate_schema(csv, jsonl):
    df = pd.read_csv(csv)
    # Extract the schema information
    schema = df.dtypes.to_dict()
    schema_list = []
    for key, value in schema.items():
        if " " in key:
            new_key = key.replace(" ", "_") # Remove the space in the column name
        else:
            new_key = key
        if value == "object":
            value = "TEXT"
            unique_values = df[key].dropna().unique().tolist()
            if not bool(re.search(r'[A-Za-z]', unique_values[0])):
                schema_list.append({"Column Name": new_key, "Column Type": value})
            else:
                schema_list.append({"Column Name": new_key, "Column Type": value, "Sample Values": unique_values})
        else:
            value = "NUMERIC"
            schema_list.append({"Column Name": new_key, "Column Type": value})


    # Save the schema information to a JSON file
    with open(jsonl, "w") as f:
        for item in schema_list:
            json.dump(item, f)
            f.write("\n")

if __name__ =='__main__':
    generate_schema()
