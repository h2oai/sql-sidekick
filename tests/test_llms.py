import os
from pathlib import Path

import pytest
from sidekick.prompter import ask, db_setup
from sidekick.query import SQLGenerator
from sidekick.schema_generator import generate_schema
from sidekick.utils import setup_dir

base_path = (Path(__file__).parent / "../").resolve()
base_path = "."
cache_path = f"{base_path}/var/lib/tmp"
setup_dir(base_path)

HOST_NAME = "localhost"
USER_NAME = "sqlite"
PASSWORD = "abc"
DB_NAME = "query_test_db"
PORT = "5432"

data_path = "./examples/demo/sleep_health_and_lifestyle_dataset.csv"
# Replace table_name as needed
table_name = "sleep_health_and_lifestyle"
_, table_info_path = generate_schema(data_path, f"{cache_path}/{table_name}_table_info.jsonl")
# Set DB and table to test
# Set add_sample=False if no need to add rows to the table (default: = True)
# Initialize DB
if Path(f"{base_path}/db/sqlite/{DB_NAME}.db").exists():
    os.remove(f"{base_path}/db/sqlite/{DB_NAME}.db")

_, err = db_setup(
                db_name=DB_NAME,
                hostname=HOST_NAME,
                user_name=USER_NAME,
                password=PASSWORD,
                port=PORT,
                table_info_path=table_info_path,
                table_samples_path=data_path,
                table_name=table_name,
                local_base_path=base_path
            )

# Currently, testing the remote model generation
def test_remote_model():
   # 1.
    input_q = """What is the average sleep duration for each gender?"""

    # or something similar
    expected_1 = """'Male', 7.036507936507934"""
    expected_2 = """'Female', 7.229729729729729"""

    result, _ar, error = ask(
        question=input_q,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder-34b-alpha",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=False,
        self_correction=True
    )

    assert expected_1 in result
    assert expected_2 in result

    input_q = """What are the most common occupations among individuals in the dataset?"""

    # or something similar
    expected = None

    result, _ar, error = ask(
        question=input_q,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder-34b-alpha",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=False,
        self_correction=True
    )

    assert result is not expected
