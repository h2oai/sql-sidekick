import logging
import os
import sys
from pathlib import Path

import pytest
import sqlglot
import toml
from sidekick.prompter import ask, db_setup
from sidekick.query import SQLGenerator
from sidekick.schema_generator import generate_schema
from sidekick.utils import setup_dir

LOGGER = logging.getLogger(__name__)

def test_self_correction():
    input_sql = """
SELECT "age_bucket",
       AVG("sleep_duration") AS "average_sleep_duration"
FROM
  (SELECT "age" AS "age_bucket"
   FROM "sleep_health_and_lifestyle_study"
   WHERE LOWER('Gender') LIKE CONCAT('%like%', '%Female,Male%')
     AND LOWER('Occupation') LIKE '%Accountant,Doctor,Engineer,Lawyer,Manager,Nurse,Sales Representative,Salesperson,Scientist,Software Engineer,Teacher%'
     AND LOWER('BMI_Category') LIKE '%Normal,Normal Weight,Obese,Overweight%'
     AND LOWER('Blood_Pressure') LIKE '%115/75,%115/78,%117/76,%118/75,%118/76,%119/77%'
     AND LOWER('Sleep_Disorder') LIKE '%Insomnia,Sleep Apnea%'
   GROUP BY "age") AS "age_buckets"
JOIN "sleep_health_and_lifestyle_study" ON "age_buckets"."age_bucket" = "sleep_health_and_lifestyle_study"."age"
GROUP BY "age_buckets"."age_bucket"
ORDER BY "age_buckets"."age_bucket" NULLS LAST
LIMIT 100
    """

    # Initialize
    # Set DB and table to test

    base_path = (Path(__file__).parent / "../").resolve()
    base_path = "./"
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

    # Set add_sample=False if no need to add rows to the table (default: = True)
    response, err = db_setup(
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

    result = None
    try:
        input_q = sqlglot.transpile(input_sql, identify=True, write='sqlite')[0]
        # Execute SQL to catch runtime error in debug mode, no generation on the input text

        question = f"Execute SQL:\n{input_q}"
        print("Execute SQL")
        res = ask(
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
    )
    except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
        _, _, ex_traceback = sys.exc_info()
        print(f"Attempting to fix syntax error ...,\n {e}")
        env_url = os.environ["RECOMMENDATION_MODEL_REMOTE_URL"]
        env_key = os.environ["H2OAI_KEY"]
        try:
            sql_g = SQLGenerator(
            None,
            None,
            model_name=None,
            job_path=None,
            data_input_path=None,
            sample_queries_path=None,
            is_regenerate_with_options=None,
            is_regenerate=None,
            db_dialect="sqlite")
            print(f"here ... {res}")
            corr_sql =  sql_g.self_correction(res, error_msg=ex_traceback, remote_url=env_url, client_key=env_key)
            result = sqlglot.transpile(corr_sql, identify=True, write='sqlite')[0]
        except Exception as se:
            print(f"We did the best we could, there might be still be some error:\n {se}")
            print(f"Realized query so far:\n {result}")
    print(result)
    assert result != input_sql
