import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import openai
import sqlglot
import torch
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, GPTSQLStructStoreIndex, LLMPredictor, ServiceContext, SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder
from sidekick.configs.prompt_template import DEBUGGING_PROMPT, NSQL_QUERY_PROMPT, QUERY_PROMPT, TASK_PROMPT
from sidekick.logger import logger
from sidekick.utils import filter_samples, read_sample_pairs, remove_duplicates
from sqlalchemy import create_engine
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        sample_queries_path: str = "./samples.csv",
        job_path: str = "../var/lib/tmp/data",
    ):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.sql_database = SQLDatabase(self.engine)
        self.similarity_model = None
        self.context_builder = None
        self.data_input_path = _check_file_info(data_input_path)
        self.sample_queries_path = sample_queries_path
        self.path = job_path
        self._data_info = None
        self._tasks = None
        self.openai_key = openai_key
        self.content_queries = None
        self.model = None  # Used for local LLMs
        self.tokenizer = None  # Used for local tokenizer

    def load_column_samples(self, tables: list):
        # TODO: Maybe we add table name as a member variable
        #  Load column values if they exists
        examples = {}
        for _t in tables:
            f_p = f"{self.path}/var/lib/tmp/data/{_t}_column_values.json"
            with open(f_p, "r") as f:
                examples[_t] = json.load(f)
        return examples

    def build_index(self, persist: bool = True):
        # Below re-assignment of the OPENAI API key is weird but without that, it throws an error.
        if self.openai_key:
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
            logger.info(f"Using QnA samples from path {self.sample_queries_path}")
            new_context_queries = read_sample_pairs(self.sample_queries_path, "h2ogpt-sql")
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

    def generate_response(
        self, context_container, sql_index, input_prompt, attempt_fix_on_error: bool = True, _dialect: str = "sqlite"
    ):
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
                    logger.info(f"Attempting to fix syntax error ...,\n {se}")
                    system_prompt = DEBUGGING_PROMPT["system_prompt"].format(_dialect=_dialect)
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
            filtered_context, _ = (
                filter_samples(input_question, updated_context, m_path, threshold=0.9)
                if len(updated_context) > 1
                else updated_context
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
            task_list = self._query_tasks(input_question, data_info, _queries, table_names)
            with open(f"{self.path}/var/lib/tmp/data/tasks.txt", "w") as f:
                f.write(task_list)
            return task_list
        except Exception as se:
            raise se

    def generate_sql(
        self, table_names: list, input_question: str, _dialect: str = "sqlite", model_name: str = "h2ogpt-sql"
    ):
        context_file = f"{self.path}/var/lib/tmp/data/context.json"
        additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}
        context_queries = self.content_queries

        table_context_dict = {str(table_names[0]).lower(): str(additional_context).lower()}
        self.context_builder = SQLContextContainerBuilder(self.sql_database, context_dict=table_context_dict)

        if model_name != "h2ogpt-sql":
            _tasks = self.task_formatter(self._tasks)

            # TODO: The need to pass data info again could be eliminated if Task generation becomes more consistent and accurate.
            query_str = QUERY_PROMPT.format(
                _dialect=_dialect,
                _data_info=self._data_info,
                _question=input_question,
                _table_name=table_names,
                _sample_queries=context_queries,
                _tasks=_tasks,
            )

            # Reference: https://github.com/jerryjliu/llama_index/issues/987
            llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(temperature=0.5, model_name=model_name))
            service_context_gpt3 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt3, chunk_size_limit=512)

            table_schema_index = self.build_index(persist=False)
            self.context_builder.query_index_for_context(table_schema_index, query_str, store_context_str=True)
            context_container = self.context_builder.build_context_container()

            index = GPTSQLStructStoreIndex(
                [], sql_database=self.sql_database, table_name=table_names, service_context=service_context_gpt3
            )
            res = self.generate_response(context_container, sql_index=index, input_prompt=query_str)
            try:
                # Check if `SQL` is formatted ---> ``` SQL_text ```
                if "```" in str(res):
                    res = (
                        str(res)
                        .split("```", 1)[1]
                        .split(";", 1)[0]
                        .strip()
                        .replace("```", "")
                        .replace("sql\n", "")
                        .strip()
                    )
                else:
                    res = str(res).split("Explanation:", 1)[0].strip()
                sqlglot.transpile(res)
            except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
                logger.info("We did the best we could, there might be still be some error:\n")
                logger.info(f"Realized query so far:\n {res}")
        else:
            # Load h2oGPT.NSQL model
            device = {"": 0} if torch.cuda.is_available() else "cpu"
            # https://github.com/pytorch/pytorch/issues/52291
            _load_in_8bit = False if "cpu" in device else True
            if self.model is None:
                self.tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B", device_map=device)
                self.model = AutoModelForCausalLM.from_pretrained(
                    "NumbersStation/nsql-6B", device_map=device, load_in_8bit=_load_in_8bit
                )

            # TODO Update needed for multiple tables
            columns_w_type = (
                self.context_builder.full_context_dict[table_names[0]].split(":")[2].split("and")[0].strip()
            )

            data_samples_list = self.load_column_samples(table_names)

            _context = {
                "if patterns like 'current time' or 'now' occurs in question": "always use NOW() - INTERVAL",
                "if patterns like 'total number', or 'List' occurs in question": "always use DISTINCT",
            }
            m_path = f"{self.path}/var/lib/tmp/.cache/models"
            filtered_context, self.similarity_model = filter_samples(
                model_obj=self.similarity_model,
                input_q=input_question,
                probable_qs=list(_context.keys()),
                model_path=m_path,
                threshold=0.90,
            )
            logger.debug(f"Filter Context: {filtered_context}")

            contextual_context = []
            for _item in filtered_context:
                _val = _context.get(_item, None)
                if _val:
                    contextual_context.append(f"{_item}: {_val}")

            logger.info("Filtering Question/Query pairs ...")
            context_queries: list = self.update_context_queries()
            logger.info(f"Number of context queries found: {len(context_queries)}")

            # Remove duplicates from the context queries
            m_path = f"{self.path}/var/lib/tmp/.cache/models"
            # duplicates_idx = remove_duplicates(context_queries, m_path, similarity_model=self.similarity_model)
            # updated_context = np.delete(np.array(context_queries), duplicates_idx).tolist()

            # Filter closest samples to the input question, threshold = 0.9
            filtered_context, _ = (
                filter_samples(
                    input_q=input_question,
                    probable_qs=context_queries,
                    model_path=m_path,
                    model_obj=self.similarity_model,
                    threshold=0.9,
                )
                if len(context_queries) > 1
                else (context_queries, None)
            )
            logger.info(f"Number of possible contextual queries to question: {len(filtered_context)}")
            # If QnA pairs > 5, we keep top 5 for focused context
            _samples = filtered_context
            if len(filtered_context) > 3:
                _samples = filtered_context[0:3][::-1]

            qna_samples = "\n".join(_samples)

            contextual_context_val = ", ".join(contextual_context)
            column_names = columns_w_type.strip().split(",")
            clmn_names = [i.split("(")[0].strip() for i in column_names]

            context_columns = []
            if len(_samples) > 2:
                # Check for the columns in the QnA samples provided, if exists keep them
                context_columns = [_c for _c in clmn_names if _c.lower().strip() in qna_samples.lower()]
                if len(context_columns) > 0:
                    contextual_data_samples = [
                        _d
                        for _cc in context_columns
                        for _d in data_samples_list[table_names[0]]
                        if _cc.lower() in _d.lower()
                    ]
                    data_samples_list = contextual_data_samples
            relevant_columns = context_columns if len(context_columns) > 0 else clmn_names
            _column_info = ", ".join(relevant_columns)

            logger.debug(f"Relevant sample column values: {data_samples_list}")
            _table_name = ", ".join(table_names)
            query = NSQL_QUERY_PROMPT.format(
                table_name=_table_name,
                column_info=_column_info,
                data_info_detailed=data_samples_list,
                sample_queries=qna_samples,
                context=contextual_context_val,
                question_txt=input_question,
            )

            logger.debug(f"Query Text:\n {query}")
            inputs = self.tokenizer([query], return_tensors="pt")
            input_length = 1 if self.model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            logger.info(f"Context length: {input_length}")
            # Generate SQL
            random_seed = random.randint(0, 50)
            torch.manual_seed(random_seed)

            # Greedy search for quick response
            self.model.eval()
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            output = self.model.generate(
                **inputs.to(device_type),
                max_new_tokens=300,
                temperature=0.5,
                output_scores=True,
                return_dict_in_generate=True,
            )

            generated_tokens = output.sequences[:, input_length:]
            _res = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            # Below is a pre-caution in-case of an error in table name during generation
            # COLLATE NOCASE is used to ignore case sensitivity, this might be specific to sqlite
            _temp = _res.replace("table_name", table_names[0]).split(";")[0]
            res = "SELECT" + _temp + ";"

            # Validate the generate SQL for parsing errors, along with dialect specific validation
            # Note: Doesn't do well with handling date-time conversions
            # e.g.
            # sqlite: SELECT DATETIME(MAX(timestamp), '-5 minute') FROM demo WHERE isin_id = 'VM88109EGG92'
            # postgres: SELECT MAX(timestamp) - INTERVAL '5 minutes' FROM demo where isin_id='VM88109EGG92'
            # Reference ticket: https://github.com/tobymao/sqlglot/issues/2011
            result = res
            try:
                result = sqlglot.transpile(res, identify=True, write="sqlite")[0]
            except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
                logger.info("We did the best we could, there might be still be some error:\n")
                logger.info(f"Realized query so far:\n {res}")
        return result

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
