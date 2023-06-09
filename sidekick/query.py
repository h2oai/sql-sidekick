import json
import os
import sys
from pathlib import Path

import numpy as np
import openai
import sqlglot
import toml
from langchain import OpenAI
from llama_index import (GPTSimpleVectorIndex, GPTSQLStructStoreIndex,
                         LLMPredictor, ServiceContext, SQLDatabase)
from llama_index.indices.struct_store import SQLContextContainerBuilder
from loguru import logger
from sidekick.configs.prompt_template import (DEBUGGING_PROMPT, QUERY_PROMPT,
                                              TASK_PROMPT)
from sidekick.examples.sample_data import sample_values, samples_queries
from sidekick.utils import remove_duplicates
from sqlalchemy import create_engine

logger.remove()
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/sidekick/configs/.env.toml")
logger.add(sys.stderr, level=env_settings['LOGGING']['LOG-LEVEL'])


class SQLGenerator:
    def __init__(self, db_url: str, openai_key: str = None, path: str = "../var/lib/tmp/data"):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.sql_database = SQLDatabase(self.engine)
        self.context_builder = None
        self.path = path
        self._tasks = None
        self.openai_key = openai_key
        self.content_queries = None

    def build_index(self, persist: bool = True):
        # Below re-assignment of the OPENAI API key is weird but without that, it throws an error.
        os.environ["OPENAI_API_KEY"] = self.openai_key
        openai.api_key = self.openai_key

        table_schema_index = self.context_builder.derive_index_from_context(
            GPTSimpleVectorIndex,
        )
        if persist:
            table_schema_index.save_to_disk(f"{self.path}/sql_index_check.json")
        return table_schema_index

    def update_context_queries(self):
        # Read the history file and update the context queries
        new_context_queries = samples_queries
        history_file = f"{self.path}/var/lib/tmp/data/history.jsonl"
        if Path(history_file).exists():
            with open(history_file, "r") as in_file:
                for line in in_file:
                    # Format:
                    # """
                    # # query:
                    # # answer:
                    # """
                    query = json.loads(line)["Query"]
                    response = json.loads(line)["Answer"]
                    _new_samples = f"""# query: {query}\n# answer: {response}"""
                    new_context_queries.append(_new_samples)
        return new_context_queries

    def _query_tasks(self, question_str, data_info, sample_queries, table_name: list):
        try:
            context_file = f"{self.path}/var/lib/tmp/data/context.json"
            additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}

            system_prompt = TASK_PROMPT["system_prompt"]
            user_prompt = TASK_PROMPT["user_prompt"].format(
                _table_name=table_name,
                _data_info=data_info,
                _sample_queries=sample_queries,
                _context=str(additional_context).lower(),
                _question_str=question_str,
            )
            # Role and content
            query_txt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            logger.debug(f"Query Text:\n {query_txt}")

            # TODO ADD local model
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=query_txt,
            )
            res = completion.choices[0].message["content"]
            return res
        except Exception as se:
            _, ex_value, _ = sys.exc_info()
            res = ex_value.statement
            return res

    def generate_response(self, context_container, sql_index, input_prompt, attempt_fix_on_error: bool = True):
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
                    logger.info(f"Attempting to heal ...,\n {se}")
                    system_prompt = DEBUGGING_PROMPT["system_prompt"]
                    user_prompt = DEBUGGING_PROMPT["user_prompt"].format(ex_traceback=ex_traceback, qry_txt=qry_txt)
                    # Role and content
                    query_msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages=query_msg,
                    )
                    res = completion.choices[0].message["content"]
                    if "SELECT" not in res:
                        res = qry_txt
                    return res
                except Exception as se:
                    # Another exception occurred, return the original SQL
                    res = qry_txt
                    return res

    def generate_tasks(self, table_names: list, input_question: str):
        try:
            # Step 1: Given a question, generate tasks to possibly answer the question and persist the result -> tasks.txt
            # Step 2: Append task list to 'query_prompt_template', generate SQL code to answer the question and persist the result -> sql.txt
            context_queries: list = self.update_context_queries()

            # Remove duplicates from the context queries
            m_path = f"{self.path}/var/lib/tmp/.cache/models"
            duplicates_idx = remove_duplicates(context_queries, m_path)
            updated_context = np.delete(np.array(context_queries), duplicates_idx).tolist()

            _queries = "\n".join(updated_context)
            self.content_queries = _queries
            task_list = self._query_tasks(input_question, sample_values, _queries.lower(), table_names)
            with open(f"{self.path}/var/lib/tmp/data/tasks.txt", "w") as f:
                f.write(task_list)
            return task_list
        except Exception as se:
            raise se

    def generate_sql(self, table_name: list, input_question: str, _dialect: str = "postgres"):
        _tasks = self.task_formatter(self._tasks)
        context_file = f"{self.path}/var/lib/tmp/data/context.json"
        additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}

        # Attempt updating in-case additional context is provided
        context_queries = None
        if self.content_queries:
            context_queries = self.content_queries
        else:
            context_queries = self.update_context_queries()
        query_str = QUERY_PROMPT.format(
            dialect=_dialect,
            _question=input_question.lower(),
            table_name=table_name,
            _sample_queries=context_queries.lower(),
            _tasks=_tasks.lower(),
        )

        table_context_dict = {str(table_name[0]).lower(): str(additional_context).lower()}
        self.context_builder = SQLContextContainerBuilder(self.sql_database, context_dict=table_context_dict)

        table_schema_index = self.build_index(persist=False)
        self.context_builder.query_index_for_context(table_schema_index, query_str, store_context_str=True)
        context_container = self.context_builder.build_context_container()

        # Reference: https://github.com/jerryjliu/llama_index/issues/987
        llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003"))
        service_context_gpt3 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt3, chunk_size_limit=512)

        index = GPTSQLStructStoreIndex(
            [], sql_database=self.sql_database, table_name=table_name, service_context=service_context_gpt3
        )
        res = self.generate_response(context_container, sql_index=index, input_prompt=query_str)
        try:
            # Check if `SQL` is formatted ---> ``` SQL_text ```
            if "```" in str(res):
                res = (
                    str(res).split("```", 1)[1].split(";", 1)[0].strip().replace("```", "").replace("sql\n", "").strip()
                )
            sqlglot.transpile(res)
        except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
            logger.info("We did the best we could, there might be still be some error:\n")
            logger.info(f"Realized query so far:\n {res}")
        return res

    def task_formatter(self, input_task: str):
        # Generated format
        """
        Tasks:
        1. Generate a SELECT query to display all columns of the {selected tables}.
        2. Infer the return type of the question as a description of the table schema.
        3. Final output: Return the table schema for the selected table.
        """

        # Converted format
        """
        # 1. Generate a SELECT query to display all columns of the {selected tables}.
        # 2. Infer the return type of the question as a description of the table schema.
        """
        _res = input_task.split("\n")
        start_index = 1 if "Tasks" in _res[0] else 0
        res = "\n".join([f"# {i}" for i in _res[start_index:]])  # Skip the first line
        return res
