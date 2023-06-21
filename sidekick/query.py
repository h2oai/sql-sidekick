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
from sidekick.configs.prompt_template import (DEBUGGING_PROMPT, QUERY_PROMPT,
                                              TASK_PROMPT)
from sidekick.logger import logger
from sidekick.utils import csv_parser, filter_samples, remove_duplicates
from sqlalchemy import create_engine


def _check_file_info(file_path: str):
    if file_path is not None and Path(file_path).exists():
        logger.info(f"Using information info from path {file_path}")
        return file_path
    else:
        logger.info("Required info not found, provide a path for table information and try again")
        raise FileNotFoundError(f"Table info not found at {file_path}")


class SQLGenerator:
    def __init__(
        self,
        db_url: str,
        openai_key: str = None,
        data_input_path: str = "./table_info.jsonl",
        samples_queries: str = "./samples.csv",
        job_path: str = "../var/lib/tmp/data",
    ):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.sql_database = SQLDatabase(self.engine)
        self.context_builder = None
        self.data_input_path = _check_file_info(data_input_path)
        self.sample_queries_path = samples_queries
        self.path = job_path
        self._data_info = None
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
        # Check if seed samples were provided
        new_context_queries = []
        if self.sample_queries_path is not None and Path(self.sample_queries_path).exists():
            logger.info(f"Using samples from path {self.sample_queries_path}")
            new_context_queries = csv_parser(self.sample_queries_path)
            # cache the samples for future use
            with open(f"{self.path}/var/lib/tmp/data/queries_cache.json", "w") as f:
                json.dump(new_context_queries, f, indent=2)
        elif self.sample_queries_path is None and Path(f"{self.path}/var/lib/tmp/data/queries_cache.json").exists():
            logger.info(f"Using samples from cache")
            with open(f"{self.path}/var/lib/tmp/data/queries_cache.json", "r") as f:
                new_context_queries = json.load(f)
        # Read the history file and update the context queries
        history_file = f"{self.path}/var/lib/tmp/data/history.jsonl"
        try:
            if Path(history_file).exists():
                with open(history_file, "r") as in_file:
                    for line in in_file:
                        # Format:
                        # """
                        # # query:
                        # # answer:
                        # """
                        if line.strip():
                            data = json.loads(line)
                            if "Query" in data and "Answer" in data:
                                query = data["Query"]
                                response = data["Answer"]
                            _new_samples = f"""# query: {query}\n# answer: {response}"""
                            new_context_queries.append(_new_samples)
        except ValueError as ve:
            logger.error(f"Error in reading history file: {ve}")
            pass
        return new_context_queries

    def _query_tasks(self, question_str, data_info, sample_queries, table_name: list):
        try:
            context_file = f"{self.path}/var/lib/tmp/data/context.json"
            additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}

            system_prompt = TASK_PROMPT["system_prompt"]
            user_prompt = TASK_PROMPT["user_prompt"].format(
                _table_name=",".join(table_name),
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
            res = ex_value.statement if ex_value.statement else None
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
            logger.info(f"Number of context queries found: {len(context_queries)}")

            # Remove duplicates from the context queries
            m_path = f"{self.path}/var/lib/tmp/.cache/models"
            duplicates_idx = remove_duplicates(context_queries, m_path)
            updated_context = np.delete(np.array(context_queries), duplicates_idx).tolist()

            # Filter closest samples to the input question, threshold = 0.45
            filtered_context = (
                filter_samples(input_question, updated_context, m_path) if len(updated_context) > 1 else updated_context
            )
            logger.info(f"Number of possible contextual queries to question: {len(filtered_context)}")
            _queries = "\n".join(filtered_context)
            self.content_queries = _queries

            # data info
            input_file = self.data_input_path
            data_info = ""
            with open(input_file, "r") as in_file:
                for line in in_file:
                    if line.strip():
                        data = json.loads(line)
                        data_info += "\n" + json.dumps(data)
            self._data_info = data_info
            task_list = self._query_tasks(input_question, data_info, _queries.lower(), table_names)
            with open(f"{self.path}/var/lib/tmp/data/tasks.txt", "w") as f:
                f.write(task_list)
            return task_list
        except Exception as se:
            raise se

    def generate_sql(self, table_name: list, input_question: str, _dialect: str = "postgres"):
        _tasks = self.task_formatter(self._tasks)
        context_file = f"{self.path}/var/lib/tmp/data/context.json"
        additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}

        context_queries = self.content_queries
        # TODO: The need to pass data info again could be eliminated if Task generation becomes more consistent and accurate.
        query_str = QUERY_PROMPT.format(
            _dialect=_dialect,
            _data_info=self._data_info,
            _question=input_question.lower(),
            _table_name=table_name,
            _sample_queries=context_queries,
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
