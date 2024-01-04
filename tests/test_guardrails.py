import os
from pathlib import Path

import pytest
import sqlglot
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

def test_no_error():
    input_q = """SELECT * FROM sleep_health_and_lifestyle_study;"""

    result = None
    question = f"Execute SQL:\n{input_q}"

    result, _ar, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder2",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=False
    )

    assert 'malicious patterns' not in str(result)

def test_blind_select_injection():
    input_q = """
    SELECT * FROM sleep_health_and_lifestyle_study WHERE person_id = 105 OR 1=1;"
    """

    result = None
    question = f"Execute SQL:\n{input_q}"
    #1. Self correction is disabled
    result, _ar, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder2",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=False
    )

    assert 'malicious patterns' in str(result)
    assert error is None

def test_drop_injection():
    input_q = ["""
    DROP sleep_health_and_lifestyle_study;--"
    """,
    """DROP sleep_health_and_lifestyle_study;
    """,
    """DROP sleep_health_and_lifestyle_study;#""",
    """10; DROP TABLE sleep_health_and_lifestyle_study /*"""
    ]


    #1. Self correction is disabled
    for _item in input_q:
        result = None
        question = f"Execute SQL:\n{_item}"
        result, _, error = ask(
            question=question,
            table_info_path=table_info_path,
            sample_queries_path=None,
            table_name=table_name,
            is_command=False,
            model_name="h2ogpt-sql-sqlcoder2",
            is_regenerate=False,
            is_regen_with_options=False,
            execute_query=True,
            local_base_path=base_path,
            debug_mode=True,
            self_correction=False
        )
        assert 'malicious patterns' in str(result)
        assert 'SQL keywords does not start with SELECT' in str(result)


def test_stacked_queries():
    input_q = """SELECT * FROM sleep_health_and_lifestyle_study; DROP sleep_health_and_lifestyle_study"""

    result = None
    question = f"Execute SQL:\n{input_q}"

    result, _ar, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder2",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=False
    )

    assert 'malicious patterns' in str(result)
    assert 'drop' in str(result)
