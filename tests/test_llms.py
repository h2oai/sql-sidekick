import os
from pathlib import Path

import pytest
from sidekick.prompter import ask, db_setup
from sidekick.query import SQLGenerator
from sidekick.schema_generator import generate_schema
from sidekick.utils import generate_text_embeddings, setup_dir
from sklearn.metrics.pairwise import cosine_similarity

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

def compute_similarity_score(x1: str, x2:str):
    m_path = f"{base_path}/models/sentence_transformers/"
    _embedding1 = generate_text_embeddings(m_path, x=[x1, x2])
    _embedding2 = generate_text_embeddings(m_path, x=[x2])
    similarities_score = cosine_similarity(_embedding1.astype(float), _embedding2.astype(float))
    return similarities_score


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
def test_basic_access():
    # 1.
    input_q = """What is the average sleep duration for each gender?"""
    expected_1 = "Male"
    expected_2 = "Female"

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
        guardrails=False,
        self_correction=True
    )

    assert expected_1 in str(result)
    assert expected_2 in str(result)


def test_input1():
    # 2.
    input_q = """What are the most common occupations among individuals in the dataset?"""
    expected_value = str([('Nurse', 73), ('Doctor', 71), ('Engineer', 63), ('Lawyer', 47), ('Teacher', 40), ('Accountant', 37), ('Salesperson', 32), ('Software Engineer', 4), ('Scientist', 4), ('Sales Representative', 2), ('Manager', 1)])
    expected_sql = """SELECT "Occupation", COUNT(*) AS "frequency" FROM "sleep_health_and_lifestyle" GROUP BY "Occupation" ORDER BY "frequency" DESC LIMIT 10
    """

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
        guardrails=False,
        self_correction=True
    )
    _generated_sql = str(result[1].split("``` sql\n")[1])
    _runtime_value = str(result[4])

    _syntax_score = compute_similarity_score(expected_sql, _generated_sql)
    _execution_val_score = compute_similarity_score(expected_value, _runtime_value)
    # compute similarity score
    assert _syntax_score[0][0] > 0.9
    assert _execution_val_score[0][0] > 0.85
