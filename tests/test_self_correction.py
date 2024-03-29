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
DB_NAME = "query_test"
PORT = "5432"

data_path = "./examples/demo/sleep_health_and_lifestyle_dataset.csv"
# Replace table_name as needed
table_name = "sleep_health_and_lifestyle"
_, table_info_path = generate_schema(data_path=data_path, output_path=f"{cache_path}/{table_name}_table_info.jsonl")
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

def test_input1():
    input_q = """
    SELECT "age", AVERAGE("sleep_duration") AS "average_sleep_duration" group by "age"
        FROM "sleep_health_and_lifestyle"
    """

    result = None
    question = f"Execute SQL:\n{input_q}"
    # 1. Self correction is disabled
    _, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name='h2ogpt-sql-sqlcoder-34b-alpha',
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=False
    )
    assert error is not None

    # 2. Self correction enabled
    result, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name='h2ogpt-sql-sqlcoder-34b-alpha',
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=True
    )
    assert result != input_q
    assert error is None

def test_input2():
    input_q = """
SELECT "age_bucket",
       AVG("sleep_duration") AS "average_sleep_duration"
FROM
  (SELECT "age" AS "age_bucket"
   FROM "sleep_health_and_lifestyle"
   WHERE LOWER('Gender') LIKE CONCAT('%like%', '%Female,Male%')
     AND LOWER('Occupation') LIKE '%Accountant,Doctor,Engineer,Lawyer,Manager,Nurse,Sales Representative,Salesperson,Scientist,Software Engineer,Teacher%'
     AND LOWER('BMI_Category') LIKE '%Normal,Normal Weight,Obese,Overweight%'
     AND LOWER('Blood_Pressure') LIKE '%115/75,%115/78,%117/76,%118/75,%118/76,%119/77%'
     AND LOWER('Sleep_Disorder') LIKE '%Insomnia,Sleep Apnea%'
   GROUP BY "age") AS "age_buckets"
JOIN "sleep_health_and_lifestyle" ON "age_buckets"."age_bucket" = "sleep_health_and_lifestyle"."age"
GROUP BY "age_buckets"."age_bucket"
ORDER BY "age_buckets"."age_bucket" NULLS LAST
LIMIT 100
"""

    result = None
    question = f"Execute SQL:\n{input_q}"
    # 1. Self correction is disabled
    _, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name=None,
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=False
    )

    assert error is not None

    # 2. Self correction enabled
    result, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder-34b-alpha",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        self_correction=True
    )
    assert result != input_q
    assert error is None

@pytest.mark.parametrize("input_q, debugger, base_model", [("""SELECT CONCAT("age", " ", "heart_rate") AS "age_heart_rate" FROM "sleep_health_and_lifestyle" ORDER BY "age_heart_rate" DESC LIMIT 100
    """, "h2oai/h2ogpt-4096-llama2-70b-chat", "h2ogpt-sql-sqlcoder-34b-alpha"),
("""SELECT CONCAT("age", " ", "heart_rate") AS "age_heart_rate" FROM "sleep_health_and_lifestyle" ORDER BY "age_heart_rate" DESC LIMIT 100
    """, "h2oai/h2ogpt-4096-llama2-70b-chat", "h2ogpt-sql-sqlcoder-34b-alpha"),
("""SELECT CONCAT("age", " ", "heart_rate") AS "age_heart_rate" FROM "sleep_health_and_lifestyle" ORDER BY "age_heart_rate" DESC LIMIT 100
    """, "h2oai/h2ogpt-4096-llama2-70b-chat", "h2ogpt-sql-sqlcoder-7b-2"),
("""SELECT CONCAT("age", " ", "heart_rate") AS "age_heart_rate" FROM "sleep_health_and_lifestyle" ORDER BY "age_heart_rate" DESC LIMIT 100
    """, "gpt-3.5-turbo", "h2ogpt-sql-sqlcoder-7b-2")])
def test_input3(input_q, debugger, base_model):
    # There is no CONCAT function in SQLite
    os.environ["SELF_CORRECTION_MODEL"] = debugger
    question = f"Execute SQL:\n{input_q}"
    res, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name=base_model,
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=True,
        guardrails=False,
        self_correction=True
    )
    assert error == None
    assert res != None

# Fixing correlation function needs further investigation
@pytest.mark.parametrize("input_q, debugger, base_model", [
("""Correlation between sleep duration and quality of sleep""", "h2oai/h2ogpt-4096-llama2-70b-chat", "h2ogpt-sql-sqlcoder-34b-alpha"),
("""Correlation between sleep duration and quality of sleep""", "h2oai/h2ogpt-4096-llama2-70b-chat", "h2ogpt-sql-sqlcoder-7b-2"),
("""Correlation between sleep duration and quality of sleep""", "gpt-3.5-turbo", "h2ogpt-sql-sqlcoder-7b-2"),
("""Correlation between sleep duration and quality of sleep" AS "s" LIMIT 100?""", "gpt-4-8k", "h2ogpt-sql-sqlcoder-7b-2")])
def test_input4(input_q, debugger, base_model):
    # There is no CONCAT function in SQLite
    os.environ["SELF_CORRECTION_MODEL"] = debugger
    question = f"Execute SQL:\n{input_q}"
    print(f"Model Name/Debugger: {base_model}/{debugger}")
    res, _, error = ask(
        question=question,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name=base_model,
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=True,
        local_base_path=base_path,
        debug_mode=False,
        guardrails=False,
        self_correction=True
    )
    assert error == None
    assert res != None
