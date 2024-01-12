import os
from pathlib import Path
from typing import Optional

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



def run_eval(input_data_path: str, table_name: str, gold_context_path: str, model_name: str="h2ogpt-sql-sqlcoder-34b-alpha", sample_qna_path: Optional[str]=None, **kwargs):
    _, table_info_path = generate_schema(input_data_path, f"{cache_path}/{table_name}_table_info.jsonl")
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
    syntax_accuracy = []
    compare_df = pd.read_csv(gold_context_path)
    for _row in compare_df.itertuples():
        input_q = _row.question
        expected_sql = _row.answer

        # With self-correction
        result, _ar, error = ask(
            question=input_q,
            table_info_path=table_info_path,
            sample_queries_path=None,
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

        _syntax_score = compute_similarity_score(expected_sql, result)
        if _syntax_score > 0.9:
            syntax_accuracy.append(_syntax_score)
