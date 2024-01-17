import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sidekick.db_config import DBConfig
from sidekick.prompter import ask
from sidekick.query import SQLGenerator
from sidekick.schema_generator import generate_schema
from sidekick.utils import generate_text_embeddings, setup_dir
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

base_path = (Path(__file__).parent / "../").resolve()
base_path = "."
cache_path = f"{base_path}/var/lib/tmp"
setup_dir(base_path)


def compute_similarity_score(x1: str, x2:str):
    m_path = f"{base_path}/models/sentence_transformers/"
    _embedding1 = generate_text_embeddings(m_path, x=[x1, x2])
    _embedding2 = generate_text_embeddings(m_path, x=[x2])
    similarities_score = cosine_similarity(_embedding1.astype(float), _embedding2.astype(float))
    return similarities_score

# Check if below env variables are set
assert os.environ.get("DATABRICKS_HOST") is not None
assert os.environ.get("DATABRICKS_CLUSTER_ID") is not None
assert os.environ.get("DATABRICKS_TOKEN") is not None

DBConfig.dialect = "databricks"
config_args = {
    "catalog": "samples",
    "schema": "nyctaxi",
    "cluster_id": os.environ.get("DATABRICKS_CLUSTER_ID")
}
table_name = "trips" # sample table related to NYC Taxi dataset
DBConfig.table_name = table_name
column_info, table_info_path = DBConfig.get_column_info(output_path=f"{cache_path}/{table_name}_table_info.jsonl", **config_args)
print(column_info)

def test_generation_execution_correctness():
    input_q = """Compute average trip distance"""
    expected_sql = """SELECT AVG(trip_distance) AS avg_distance FROM trips"""
    _generated_sql = ""
    result, _, _ = ask(
        question=input_q,
        table_info_path=table_info_path,
        sample_queries_path=None,
        table_name=table_name,
        is_command=False,
        model_name="h2ogpt-sql-sqlcoder-34b-alpha",
        db_dialect="databricks",
        execute_db_dialect="databricks",
        is_regenerate=False,
        is_regen_with_options=False,
        execute_query=False,
        local_base_path=base_path,
        debug_mode=False,
        guardrails=False,
        self_correction=True
    )

    if result and len(result) > 0:
        _generated_sql = str(result[1].split("``` sql\n")[1])

    _syntax_score = compute_similarity_score(expected_sql, _generated_sql)
    assert _syntax_score[0][0] > 0.9
