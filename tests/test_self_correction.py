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
table_name = "test_self_correction"
_, table_info_path = generate_schema(data_path, f"{cache_path}/{table_name}_table_info.jsonl")
# Set DB and table to test
# Set add_sample=False if no need to add rows to the table (default: = True)
# Initialize DB
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
    SELECT "age_bucket", AVERAGE("sleep_duration") AS "average_sleep_duration"
        FROM "test_self_correction"
    """

    result = None
    question = f"Execute SQL:\n{input_q}"
    res, _, error = ask(
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
        debug_mode=True
    )

    if error and 'OperationalError' in error:
        env_url = os.environ["RECOMMENDATION_MODEL_REMOTE_URL"]
        env_key = os.environ["H2OAI_KEY"]

        _res = input_q
        sql_g = SQLGenerator(
            None,
            None,
            model_name=None,
            job_path=cache_path,
            data_input_path=table_info_path,
            sample_queries_path=None,
            is_regenerate_with_options=None,
            is_regenerate=None,
            debug_mode=True,
            db_dialect="sqlite"
        )
        count = 0
        while count !=2:
            try:
                print(f"Attempt: {count+1}")
                _err = error.split("\n")[0].split("Error occurred :")[1]
                corr_sql =  sql_g.self_correction(input_prompt=_res, error_msg=_err, remote_url=env_url, client_key=env_key)
                result = sqlglot.transpile(corr_sql, identify=True, write='sqlite')[0]

                _question = f"Execute SQL:\n{result}"
                _, _, error = ask(
                    question=_question,
                    table_info_path=table_info_path,
                    sample_queries_path=None,
                    table_name=table_name,
                    is_command=False,
                    model_name=None,
                    is_regenerate=False,
                    is_regen_with_options=False,
                    execute_query=True,
                    local_base_path=base_path,
                    debug_mode=True
            )
                if error and 'OperationalError' in error:
                    count += 1
                else:
                    break
            except Exception as e:
                count += 1
                print(f"Error: {e}")

        print(f"Realized query so far:\n {result}")
    assert error is None
    assert result != input_q

def test_input2():
    input_q = """
SELECT "age_bucket",
       AVG("sleep_duration") AS "average_sleep_duration"
FROM
  (SELECT "age" AS "age_bucket"
   FROM "test_self_correction"
   WHERE LOWER('Gender') LIKE CONCAT('%like%', '%Female,Male%')
     AND LOWER('Occupation') LIKE '%Accountant,Doctor,Engineer,Lawyer,Manager,Nurse,Sales Representative,Salesperson,Scientist,Software Engineer,Teacher%'
     AND LOWER('BMI_Category') LIKE '%Normal,Normal Weight,Obese,Overweight%'
     AND LOWER('Blood_Pressure') LIKE '%115/75,%115/78,%117/76,%118/75,%118/76,%119/77%'
     AND LOWER('Sleep_Disorder') LIKE '%Insomnia,Sleep Apnea%'
   GROUP BY "age") AS "age_buckets"
JOIN "test_self_correction" ON "age_buckets"."age_bucket" = "test_self_correction"."age"
GROUP BY "age_buckets"."age_bucket"
ORDER BY "age_buckets"."age_bucket" NULLS LAST
LIMIT 100
"""

    result = None
    question = f"Execute SQL:\n{input_q}"
    res, _, error = ask(
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
    )

    if error and 'OperationalError' in error:
        env_url = os.environ["RECOMMENDATION_MODEL_REMOTE_URL"]
        env_key = os.environ["H2OAI_KEY"]

        _res = input_q
        sql_g = SQLGenerator(
            None,
            None,
            model_name=None,
            job_path=cache_path,
            data_input_path=table_info_path,
            sample_queries_path=None,
            is_regenerate_with_options=None,
            is_regenerate=None,
            debug_mode=True,
            db_dialect="sqlite"
        )
        count = 0
        while count !=2:
            try:
                print(f"Attempt: {count+1}")
                _err = error.split("\n")[0].split("Error occurred :")[1]
                corr_sql =  sql_g.self_correction(input_prompt=_res, error_msg=_err, remote_url=env_url, client_key=env_key)
                result = sqlglot.transpile(corr_sql, identify=True, write='sqlite')[0]

                _question = f"Execute SQL:\n{result}"
                _, _, error = ask(
                    question=_question,
                    table_info_path=table_info_path,
                    sample_queries_path=None,
                    table_name=table_name,
                    is_command=False,
                    model_name=None,
                    is_regenerate=False,
                    is_regen_with_options=False,
                    execute_query=True,
                    local_base_path=base_path,
                    debug_mode=True
            )
                if error and 'OperationalError' in error:
                    count += 1
                else:
                    break
            except Exception as e:
                count += 1
                print(f"Error: {e}")

        print(f"Realized query so far:\n {result}")
    assert error is None
    assert result != input_q
    assert 'concat' not in result.lower()
