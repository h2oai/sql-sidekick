# 1. python3 -m venv .sidekick_venv
# 2. source .sidekick_venv/bin/activate
# 3. pip install --force-reinstall sql_sidekick-x.x.x-py3-none-any.whl (# replace x.x.x with the latest version number)

import os
from pathlib import Path
from typing import Optional

import click
import pandas as pd
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
DB_NAME = "query_eval_db"
PORT = "5432"

# Initialize DB
if Path(f"{base_path}/db/sqlite/{DB_NAME}.db").exists():
    os.remove(f"{base_path}/db/sqlite/{DB_NAME}.db")

def compute_similarity_score(x1: str, x2:str):
    m_path = f"{base_path}/models/sentence_transformers/"
    _embedding1 = generate_text_embeddings(m_path, x=[x1, x2])
    _embedding2 = generate_text_embeddings(m_path, x=[x2])
    similarities_score = cosine_similarity(_embedding1.astype(float), _embedding2.astype(float))
    return similarities_score


@click.group()
@click.version_option()
def cli():
    """For benchmarking SQL-Sidekick.
    """

@cli.command()
@click.option("--input_data_path", "-i", help="Path to dataset in .csv format")
@click.option("--table_name", "-t", help="Table name related to the supplied dataset")
@click.option("--eval_data_path", "-e", help="Path to eval dataset in .csv format")
@click.option("--model_name", "-m", default="h2ogpt-sql-sqlcoder-34b-alpha", help="Model name to use for inference")
@click.option("--sample_qna_path", "-s", default=None, help="Path to sample QnA in .csv format")
@click.option("--iterations", "-n", default=1, help="Number of iterations to run")
@click.option("--threshold", "-th", default=0.9, help="Similarity threshold")
@click.option("--kwargs", "-k", default=None, help="Additional arguments")
def run_eval(input_data_path: str, table_name: str, eval_data_path: str, model_name: str, iterations: int, threshold: float, sample_qna_path: Optional[str]=None, **kwargs):
    #  Generate schema for the supplied input data
    _, table_info_path = generate_schema(data_path=input_data_path, output_path=f"{cache_path}/{table_name}_table_info.jsonl")
    # Db setup
    _, err = db_setup(
                db_name=DB_NAME,
                hostname=HOST_NAME,
                user_name=USER_NAME,
                password=PASSWORD,
                port=PORT,
                table_info_path=table_info_path,
                table_samples_path=input_data_path,
                table_name=table_name,
                local_base_path=base_path
            )


    # read gold context
    syntax_accuracy = {}
    failures = {}
    compare_df = pd.read_csv(eval_data_path)
    count = 0
    while count < iterations:
        for _row in compare_df.itertuples():
            input_q = _row.question
            expected_sql = _row.answer

            # With self-correction
            _generated_sql = ''
            result, _, _ = ask(
                question=input_q,
                table_info_path=table_info_path,
                sample_queries_path=sample_qna_path,
                table_name=table_name,
                is_command=False,
                model_name=model_name,
                is_regenerate=False,
                is_regen_with_options=False,
                execute_query=True,
                local_base_path=base_path,
                debug_mode=False,
                guardrails=False,
                self_correction=True
            )

            if  result and len(result) > 1:
                _tmp = result[1].split("``` sql\n")
                _generated_sql = _tmp[1].strip() if len((_tmp)) > 1 else ''
            _syntax_score = compute_similarity_score(expected_sql, _generated_sql)
            if _syntax_score[0][0] > threshold:
                if input_q not in syntax_accuracy:
                    syntax_accuracy[input_q] = _syntax_score[0][0]
            else:
                if input_q not in failures:
                    failures[input_q] = (expected_sql, _generated_sql)
        count+=1
    print(f"Syntax accuracy: {float(len(syntax_accuracy)/compare_df.shape[0])}")
    print(f"Failures cases: {failures}")

if __name__ == "__main__":
    cli()
