import sys
import os
import openai
from configs.prompt_template import debugging_prompt_template, task_prompt_template
from examples.samples import sample_values
from llama_index import GPTSimpleVectorIndex, GPTSQLStructStoreIndex, SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


def build_index(db_url: str, persist: bool = True, path: str = "./var/lib/tmp/data"):
    engine = create_engine(db_url)
    sql_database = SQLDatabase(engine)
    context_builder = SQLContextContainerBuilder(sql_database)
    table_schema_index = context_builder.derive_index_from_context(
        GPTSimpleVectorIndex,
    )
    if persist:
        table_schema_index.save_to_disk(f"{path}/sql_index_check.json")
    return table_schema_index


def generate_tasks(question_str, context_info: dict, samples, table_name: str):
    keys = [table_name]
    # TODO: Throw error if context_info is not a dict.
    schema_info = list(map(context_info.get, keys))
    try:
        system_prompt = task_prompt_template["system_prompt"]
        user_prompt = task_prompt_template["user_prompt"].format(
            table_name=table_name, schema_info=schema_info, samples=samples, question_str=question_str
        )

        # Role and content
        query_txt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        # TODO ADD local model
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301",
            messages=query_txt,
        )
        res = completion.choices[0].message["content"]
        return res
    except Exception as se:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        res = ex_value.statement
        return res


def generate_response(context_container, sql_index, input_prompt, attempt_fix_on_error: bool = True):
    try:
        response = sql_index.query(input_prompt, sql_context_container=context_container)
        res = response.extra_info["sql_query"]
        return res
    except Exception as se:
        # Take the SQL and make an attempt for correction
        _, ex_value, ex_traceback = sys.exc_info()
        qry_txt = ex_value.statement
        if attempt_fix_on_error:
            try:
                # Attempt to heal with simple feedback
                # Reference: Teaching Large Language Models to Self-Debug, https://arxiv.org/abs/2304.05128
                print(f"Attempting to heal ...,\n {se}")
                system_prompt = debugging_prompt_template["system_prompt"]
                user_prompt = debugging_prompt_template["user_prompt"].format(ex_traceback, qry_txt)
                # Role and content
                query_txt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages=query_txt,
                )
                res = completion.choices[0].message["content"]
                if "SELECT" not in res:
                    res = qry_txt
                return res
            except Exception as se:
                _, ex_value, ex_traceback = sys.exc_info()
                res = ex_value.statement
                return res


def generate_sql(table_name: str, input_question: str, path: str = "./var/lib/tmp/data"):
    try:
        # Step 1: Given a question, generate tasks to possibly answer the question and persist the result -> tasks.txt
        # Step 2: Append task list to 'query_prompt_template', generate SQL code to answer the question and persist the result -> sql.txt
        host_name = "localhost"
        user_name = "postgres"
        passwd = "abc"
        db_name = "postgres"
        db_name = "querydb"
        db_url = f"postgresql+psycopg2://{user_name}:{passwd}@{host_name}/{db_name}".format(
            user_name, passwd, host_name, db_name
        )
        engine = create_engine(db_url)
        sql_database = SQLDatabase(engine)
        context_builder = SQLContextContainerBuilder(sql_database)

        c_info = context_builder.full_context_dict
        _sample_values = sample_values
        task_list = generate_tasks(input_question, c_info, _sample_values, table_name)
        with open(f"{path}/tasks.txt", "w") as f:
            f.write(task_list)
        return task_list
    except Exception as se:
        raise se
