import os
import sys

import openai
import sqlglot
from configs.prompt_template import DEBUGGING_PROMPT, QUERY_PROMPT, TASK_PROMPT
from examples.sample_data import sample_values, samples_queries
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, GPTSQLStructStoreIndex, LLMPredictor, ServiceContext, SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


class SQLGenerator:
    def __init__(self, db_url: str, openai_key: str = None):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.sql_database = SQLDatabase(self.engine)
        self.context_builder = SQLContextContainerBuilder(self.sql_database)
        self.path = "./var/lib/tmp/data"
        self._tasks = None
        self.openai_key = openai_key

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

    def _query_tasks(self, question_str, context_info: dict, samples, table_name: str):
        keys = [table_name]
        # TODO: Throw error if context_info is not a dict.
        schema_info = list(map(context_info.get, keys))
        try:
            system_prompt = TASK_PROMPT["system_prompt"]
            user_prompt = TASK_PROMPT["user_prompt"].format(
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
                    user_prompt = DEBUGGING_PROMPT["user_prompt"].format(ex_traceback, qry_txt)
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

    def generate_tasks(self, table_name: str, input_question: str, path: str = "./var/lib/tmp/data"):
        try:
            # Step 1: Given a question, generate tasks to possibly answer the question and persist the result -> tasks.txt
            # Step 2: Append task list to 'query_prompt_template', generate SQL code to answer the question and persist the result -> sql.txt
            context_builder = SQLContextContainerBuilder(self.sql_database)

            c_info = context_builder.full_context_dict
            _sample_values = sample_values
            task_list = self._query_tasks(input_question, c_info, _sample_values, table_name)
            with open(f"{path}/tasks.txt", "w") as f:
                f.write(task_list)
            return task_list
        except Exception as se:
            raise se

    def generate_sql(
        self, table_name: str, input_question: str, _dialect: str = "postgres", path: str = "./var/lib/tmp/data"
    ):
        _tasks = self.task_formatter(self._tasks)
        query_str = QUERY_PROMPT.format(
            dialect=_dialect,
            _question=input_question.lower(),
            table_name=table_name,
            _sample_queries=samples_queries.lower(),
            _tasks=_tasks.lower(),
        )

        logger.info(f"Prompt:\n {query_str}")
        table_schema_index = self.build_index(persist=False)
        self.context_builder.query_index_for_context(table_schema_index, query_str, store_context_str=True)
        context_container = self.context_builder.build_context_container()

        # Reference: https://github.com/jerryjliu/llama_index/issues/987
        llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
        service_context_gpt3 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt3, chunk_size_limit=512)

        index = GPTSQLStructStoreIndex(
            [], sql_database=self.sql_database, table_name=table_name, service_context=service_context_gpt3
        )
        res = self.generate_response(context_container, sql_index=index, input_prompt=query_str)

        try:
            # Check if `SQL` is formatted ---> ``` SQL_text ```
            if "```" in str(res):
                res = str(res).split("```", 1)[1].split("```", 1)[0]
            sqlglot.transpile(res)
        except (sqlglot.errors.ParseError, RuntimeError) as e:
            logger.info("We did the best we could, there might be still be some error:\n")
            logger.info(f"Realized query so far:\n {res}")
        return res

    def task_formatter(self, input_task: str):
        # Generated format
        #  Tasks:
        # 1. Generate a SELECT query to display all columns of the telemetry table.
        # 2. Infer the return type of the question as a description of the table schema.
        # 3. Final output: Return the table schema for the telemetry.

        # Converted format
        # # 1. Generate a SELECT query to display all columns of the telemetry table.
        # # 2. Infer the return type of the question as a description of the table schema.
        res = input_task.split(".")
        for _r in res[1:]:
            rows = "\n".join(f"# {_r}")
        return rows
