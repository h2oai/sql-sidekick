import gc
import json
import os
import random
import sys
from pathlib import Path
import requests
import numpy as np
import openai
import sqlglot
import sqlparse
import torch
import torch.nn.functional as F
from llama_index import GPTVectorStoreIndex, ServiceContext, SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder
from llama_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from llama_index.llms import OpenAI as LOpenAI
from openai import OpenAI
from sidekick.configs.prompt_template import (DEBUGGING_PROMPT,
                                              NSQL_QUERY_PROMPT, QUERY_PROMPT,
                                              STARCODER2_PROMPT, TASK_PROMPT)
from sidekick.logger import logger
from sidekick.utils import (MODEL_CHOICE_MAP_EVAL_MODE, _check_file_info,
                            is_resource_low, load_causal_lm_model,
                            load_embedding_model, make_dir, re_rank,
                            read_sample_pairs, remove_duplicates,
                            semantic_search)
from sqlalchemy import create_engine


class SQLGenerator:
    _instance = None

    def __new__(
        cls,
        db_url: str,
        openai_key: str = None,
        model_name="h2ogpt-sql-nsql-llama-2-7B-4bit",
        data_input_path: str = "./table_info.jsonl",
        sample_queries_path: str = "./samples.csv",
        db_dialect = "sqlite",
        job_path: str = "./",
        device: str = "auto",
        is_regenerate: bool = False,
        is_regenerate_with_options: bool = False,
        eval_mode = False,
        remote_model = False,
        debug_mode = False
    ):
        # TODO: If openai model then only tokenizer needs to be loaded.
        offloading = is_resource_low(model_name)
        n_gpus = torch.cuda.device_count()
        # Initially load one model at a time if the user swapped the model dynamically when GPU = 1
        # If GPU > 1, load multiple models in memory separately on each device.
        # TODO
        # Support remote model loading as an option

        if (
            offloading
            and is_regenerate_with_options
            or (n_gpus == 1 and cls._instance and cls._instance.model_name and cls._instance.model_name != model_name)
        ):
            if ("gpt-3.5" not in cls._instance.model_name or "gpt-4" not in cls._instance.model_name) and ("gpt-3.5" not in model_name or "gpt-4" not in model_name) and cls._instance.models and cls._instance.models.get(cls._instance.model_name, None):
                _name = cls._instance.model_name
                del cls._instance.models[_name]
                cls._instance.models[_name] = None
                del cls._instance.tokenizers[_name]
                cls._instance.tokenizers[_name] = None

                gc.collect()
                torch.cuda.empty_cache()
            logger.info(f"Low memory: {offloading}/ Model re-initialization: {is_regenerate_with_options}")

        if cls._instance is None or (cls._instance and hasattr(cls._instance, 'models') and not cls._instance.models.get(model_name, None)) or not hasattr(cls._instance, 'tokenizers'):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.current_temps = {}
            # Load local models only wen remote models are not selected.
            if not remote_model:
                if not debug_mode:
                    # Currently. Debug mode is using remote model
                    # This could change in future.
                    logger.info(f"Loading local model: {model_name}")
                    cls._instance.models, cls._instance.tokenizers = load_causal_lm_model(
                        model_name,
                        cache_path=f"{job_path}/models/",
                        device=device,
                        off_load=offloading,
                        re_generate=is_regenerate_with_options,
                    )
            else:
                cls._instance.models = {}
            cls._instance.model_name = "h2ogpt-sql-sqlcoder2-4bit" if not model_name else model_name
            model_embed_path = f"{job_path}/models/sentence_transformers"
            cls._instance.current_temps[cls._instance.model_name] = 0.5
            device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
            if not debug_mode:
                # Currently. Debug mode is using remote model
                # This could change in future.
                cls._instance.similarity_model = load_embedding_model(model_path=model_embed_path, device=device)
        return cls._instance

    def __init__(
        self,
        db_url: str,
        openai_key: str = None,
        model_name="h2ogpt-sql-nsql-llama-2-7B-4bit",
        data_input_path: str = "./table_info.jsonl",
        sample_queries_path: str = "./samples.csv",
        job_path: str = "./",
        device: str = "cpu",
        db_dialect = "sqlite",
        is_regenerate: bool = False,
        is_regenerate_with_options: bool = False,
        eval_mode = False,
        debug_mode = False,
        remote_model = False
    ):
        self.db_url = db_url
        self.engine = create_engine(db_url) if db_url else None
        self.sql_database = SQLDatabase(self.engine) if self.engine else None
        self.dialect = db_dialect
        self.context_builder = None
        self.data_input_path = _check_file_info(data_input_path)
        self.sample_queries_path = sample_queries_path
        self.path = job_path
        self._data_info = None
        self._tasks = None
        self.model_name = model_name
        self.openai_key = openai_key
        self.content_queries = None
        self.is_regenerate_with_options = is_regenerate_with_options
        self.is_regenerate = is_regenerate
        self.device = device
        self.table_name = None,
        self.eval_mode = eval_mode,
        self.debug_mode = debug_mode,
        self.remote_model = remote_model
        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        self.h2ogpt_client = None

    def clear(self):
        del SQLGenerator._instance
        SQLGenerator._instance = None

    def load_column_samples(self, tables: list):
        # TODO: Maybe we add table name as a member variable
        #  Load column values if they exists
        examples = {}
        for _t in tables:
            f_p = f"{self.path}/var/lib/tmp/data/{_t}_column_values.json"
            if Path(f_p).exists():
                with open(f_p, "r") as f:
                    examples[_t] = json.load(f)
        return examples

    def build_index(self, persist: bool = True):
        # Below re-assignment of the OPENAI API key is weird but without that, it throws an error.
        if self.openai_key:
            os.environ["OPENAI_API_KEY"] = self.openai_key
            openai.api_key = self.openai_key

        table_schema_index = self.context_builder.derive_index_from_context(
            GPTVectorStoreIndex,
        )
        if persist:
            table_schema_index.save_to_disk(f"{self.path}/sql_index_check.json")
        return table_schema_index

    def update_context_queries(self):
        # Check if seed samples were provided
        cache_path = f"{self.path}/var/lib/tmp/.cache/{self.table_name}/"
        new_context_queries = []
        if self.sample_queries_path is not None and Path(self.sample_queries_path).exists():
            logger.info(f"Using QnA samples from path {self.sample_queries_path}")
            new_context_queries = read_sample_pairs(self.sample_queries_path, "h2ogpt-sql")
            # cache the samples for future use
            make_dir(cache_path)
            with open(f"{cache_path}/queries_cache.json", "w") as f:
                json.dump(new_context_queries, f, indent=2)
        elif self.sample_queries_path is None and Path(f"{cache_path}/queries_cache.json").exists():
            logger.info(f"Using samples from cache")
            with open(f"{cache_path}/queries_cache.json", "r") as f:
                new_context_queries = json.load(f)
        # Read the history file and update the context queries
        history_file = f"{self.path}/var/lib/tmp/.cache/{self.table_name}/history.jsonl"
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

            MODEL_CHOICE_MAP = MODEL_CHOICE_MAP_EVAL_MODE
            m_name = MODEL_CHOICE_MAP.get(self.model_name)

            completion = self.openai_client.chat.completions.create(
                model=m_name,
                messages=query_txt,
                max_tokens=512,
                seed=42,
                temperature=0.7
            )
            res = completion.choices[0].message.content
            return res
        except Exception as se:
            _, ex_value, _ = sys.exc_info()
            res = ex_value.statement if ex_value.statement else None
            return res

    def self_correction(self, error_msg, input_query, remote_url, client_key):
        try:
            # Reference: Teaching Large Language Models to Self-Debug, https://arxiv.org/abs/2304.05128
            system_prompt = DEBUGGING_PROMPT["system_prompt"].format(dialect=self.dialect).strip()
            user_prompt = DEBUGGING_PROMPT["user_prompt"].format(ex_traceback=error_msg, qry_txt=input_query).strip()
            _response = []
            _res = input_query
            self_correction_model = os.getenv("SELF_CORRECTION_MODEL", "h2oai/h2ogpt-4096-llama2-70b-chat")
            if "h2ogpt-" in self_correction_model:
                if remote_url and client_key and remote_url != "" and client_key != "":
                    from h2ogpte import H2OGPTE
                    client = H2OGPTE(address=remote_url, api_key=client_key)
                    text_completion = client.answer_question(
                    system_prompt=system_prompt,
                    text_context_list=[],
                    question=user_prompt,
                    llm=self_correction_model)
                else:
                    logger.info(f"H2OGPTE client is not configured, attempting to use OSS H2OGPT client")
                    h2o_client_url = os.getenv("H2OGPT_BASE_URL", None)
                    h2o_client_key = os.getenv("H2OGPT_BASE_API_TOKEN", None)
                    # Make attempt to use h2ogpt client with OSS access
                    client_args = dict(base_url=h2o_client_url, api_key=h2o_client_key, timeout=20.0)
                    query_msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    h2ogpt_base_client = OpenAI(**client_args)
                    completion = h2ogpt_base_client.with_options(max_retries=3).chat.completions.create(
                                model=self_correction_model,
                                messages=query_msg,
                                max_tokens=512,
                                temperature=0.5,
                                stop="```",
                                seed=42)
                    text_completion = completion.choices[0].message
                _response = text_completion.content
            elif 'gpt-3.5' in self_correction_model.lower() or 'gpt-4' in self_correction_model.lower():
                # Check if the API key is set, else inform user
                    _self_correction_model = MODEL_CHOICE_MAP_EVAL_MODE[self_correction_model.lower()]
                    query_msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    completion = self.openai_client.chat.completions.create(
                        model=_self_correction_model,
                        messages=query_msg,
                        max_tokens=512,
                        seed=42,
                        temperature=0.7
                    )
                    _response = completion.choices[0].message.content
            else:
                raise ValueError(f"Invalid request for: {self_correction_model}")

            _response = _response.split("```sql")
            _idx = [_response.index(_r) for _r in _response if _r.lower().strip().startswith("select")]
            _res = _response[_idx[0]].split("```")[0].strip()
            if "SELECT".lower() not in _res.lower():
                _res = input_query
            result = sqlglot.transpile(_res, identify=True, write=self.dialect)[0]
            return result
        except Exception as se:
            # Another exception occurred, return the original SQL
            logger.info(f"Error in self correction: {se}")
            result = _res
            return result


    def generate_response(
        self, sql_index, input_prompt, attempt_fix_on_error: bool = True
    ):
        try:
            _sql_index = sql_index.as_query_engine()
            response = _sql_index.query(input_prompt)
            res = response.metadata["sql_query"]
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
                    system_prompt = DEBUGGING_PROMPT["system_prompt"].format(dialect=self.dialect)
                    user_prompt = DEBUGGING_PROMPT["user_prompt"].format(ex_traceback=ex_traceback, qry_txt=qry_txt)
                    # Role and content
                    query_msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    MODEL_CHOICE_MAP = MODEL_CHOICE_MAP_EVAL_MODE
                    m_name = MODEL_CHOICE_MAP.get(self.model_name, "gpt-3.5-turbo-1106")

                    completion = self.openai_client.chat.completions.create(
                        model=m_name,
                        messages=query_msg,
                        max_tokens=512,
                        seed=42,
                        temperature=0.7
                    )
                    res = completion.choices[0].message.content
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
            self.table_name = table_names[0]
            context_queries: list = self.update_context_queries()
            logger.info(f"Number of context queries found: {len(context_queries)}")

            # Remove duplicates from the context queries
            m_path = f"{self.path}/models/sentence_transformers/"
            duplicates_idx = remove_duplicates(context_queries, m_path)
            updated_context = np.delete(np.array(context_queries), duplicates_idx).tolist()

            # Filter closest samples to the input question, threshold = 0.45
            filtered_context = (
                semantic_search(
                    input_question,
                    updated_context,
                    m_path,
                    threshold=0.9,
                    is_regenerate=True if (self.is_regenerate and not self.is_regenerate_with_options) else False,
                )
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
        self,
        table_names: list,
        input_question: str,
        model_name: str = "h2ogpt-sql-sqlcoder-7b-2",
    ):
        # TODO: Update needed to support multiple tables
        table_name = str(table_names[0].replace(" ", "_")).lower()
        self.table_name = table_name
        alternate_queries = []
        describe_keywords = ["describe table", "describe", "describe table schema", "describe data"]
        enable_describe_qry = any([True for _dk in describe_keywords if _dk in input_question.lower()])

        if input_question is not None and enable_describe_qry:
            result = f"""SELECT "name" from PRAGMA_TABLE_INFO("{table_name}")"""
        else:
            context_file = f"{self.path}/var/lib/tmp/data/context.json"
            additional_context = json.load(open(context_file, "r")) if Path(context_file).exists() else {}
            table_context_dict = {table_name: str(additional_context).lower()}
            context_queries = self.content_queries
            self.context_builder = SQLContextContainerBuilder(self.sql_database, context_dict=table_context_dict)

            if model_name and "h2ogpt-sql" not in model_name:
                _tasks = self.task_formatter(self._tasks)

                # TODO: The need to pass data info again could be eliminated if Task generation becomes more consistent and accurate.
                query_str = QUERY_PROMPT.format(
                    dialect=self.dialect,
                    _data_info=self._data_info,
                    _question=input_question,
                    _table_name=table_names,
                    _sample_queries=context_queries,
                    _tasks=_tasks,
                )

                logger.debug(f"Query Text:\n {query_str}")
                # Reference: https://github.com/jerryjliu/llama_index/issues/987
                model_choices = MODEL_CHOICE_MAP_EVAL_MODE
                m_name = model_choices.get(model_name, "gpt-3.5-turbo-1106")

                llm_predictor_gpt3 = LOpenAI(temperature=0.7, model_name=m_name, max_tokens=512, seed=42)
                service_context_gpt3 = ServiceContext.from_defaults(
                    llm=llm_predictor_gpt3, chunk_size_limit=512
                )

                table_schema_index = self.build_index(persist=False)
                self.context_builder.query_index_for_context(table_schema_index, query_str, store_context_str=True)

                index = GPTSQLStructStoreIndex(
                    [], sql_database=self.sql_database, table_name=table_names, service_context=service_context_gpt3
                )

                result = self.generate_response(sql_index=index, input_prompt=query_str)
                try:
                    # Check if `SQL` is formatted ---> ``` SQL_text ```
                    if "```" in str(result):
                        res = (
                            str(result)
                            .split("```", 1)[1]
                            .split(";", 1)[0]
                            .strip()
                            .replace("```", "")
                            .replace("sql\n", "")
                            .strip()
                        )
                    else:
                        res = str(result).split("Explanation:", 1)[0].strip()
                    res = sqlglot.transpile(res, identify=True, write=self.dialect)[0]
                    result = res
                except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
                    logger.info("We did the best we could, there might be still be some error:\n")
                    logger.info(f"Realized query so far:\n {res}")
            else:
                if self.h2ogpt_client is None:
                    # Check if env variable has info about remote hosting
                    remote_h2ogpt_base_url = os.environ.get("H2OGPT_URL", None)
                    if model_name == 'h2ogpt-sql-sqlcoder-34b-alpha':
                        remote_h2ogpt_base_url = f"{remote_h2ogpt_base_url}:5000/v1"
                    elif model_name == 'h2ogpt-sql-sqlcoder-7b-2':
                        remote_h2ogpt_base_url = f"{remote_h2ogpt_base_url}:5001/v1"
                    elif model_name == 'h2ogpt-sql-nsql-llama-2-7B':
                        remote_h2ogpt_base_url = f"{remote_h2ogpt_base_url}:5002/v1"
                    else:
                        remote_h2ogpt_base_url = None
                    remote_h2ogpt_key = os.environ.get("H2OGPT_API_TOKEN", None)
                    _api_key = remote_h2ogpt_key if remote_h2ogpt_key else "EMPTY"
                    if remote_h2ogpt_base_url:
                        client_args = dict(base_url=remote_h2ogpt_base_url, api_key=_api_key, timeout=20.0)
                        self.h2ogpt_client = OpenAI(**client_args)

                # TODO Update needed for multiple tables
                columns_w_type = (
                    self.context_builder.full_context_dict[table_name]
                    .split(":")[2]
                    .split(" and foreign keys")[0]
                    .strip().replace("(", "").replace(")", "")
                )
                data_samples_list = self.load_column_samples(table_names)

                _context = {
                    "if patterns like 'current time' or 'now' occurs in question": "always use NOW() - INTERVAL",
                    "if patterns like 'total number', or 'List' occurs in question": "always use DISTINCT",
                    "detailed summary": "include min, avg, max for numeric columns",
                    "summary": "include min, avg, max for numeric columns",
                }

                m_path = f"{self.path}/models/sentence_transformers/"
                filtered_context = semantic_search(
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
                m_path = f"{self.path}/models/sentence_transformers/"
                # duplicates_idx = remove_duplicates(context_queries, m_path, similarity_model=self.similarity_model)
                # updated_context = np.delete(np.array(context_queries), duplicates_idx).tolist()

                # Filter closest samples to the input question, threshold = 0.9
                filtered_context = (
                    semantic_search(
                        input_q=input_question,
                        probable_qs=context_queries,
                        model_path=m_path,
                        model_obj=self.similarity_model,
                        threshold=0.9,
                        is_regenerate=True if (self.is_regenerate and not self.is_regenerate_with_options) else False,
                    )
                    if len(context_queries) > 1
                    else context_queries
                )
                logger.info(f"Number of possible contextual queries to question: {len(filtered_context)}")
                # If QnA pairs > 5, we keep top 5 for focused context
                # Most relevant match is closest to the generation post re-ranking
                _samples = filtered_context
                _samples = re_rank(input_question, _samples)
                if len(filtered_context) > 5:
                    _samples = filtered_context[0:5][::-1]
                    _samples = re_rank(input_question, _samples)

                qna_samples = "\n".join(_samples)

                contextual_context_val = ", ".join(contextual_context)
                column_names = columns_w_type.strip().split(",")
                clmn_names = [i.split(" ")[0].strip() for i in column_names if i]
                clmn_types = [i.split(" ")[1].strip() for i in column_names if i]
                clmn_info_map = dict(zip(clmn_names, clmn_types))

                context_columns = []
                if len(_samples) > 2:
                    # Check for the columns in the QnA samples provided, if exists keep them
                    context_columns = [_c for _c in clmn_names if _c.lower().strip() in qna_samples.lower()]

                    # To be safe, when we have more than 2 samples, we check for the column names in the question as well
                    first_pass = [_c for _c in clmn_names if _c.lower().strip() in input_question.lower().strip()]
                    _input = input_question.lower().split(" ")
                    for _c in clmn_names:
                        for _f in _c.lower().split("_"):
                            res = _f in _input
                        if res:
                            first_pass.append(_c)
                    context_columns = set(context_columns + first_pass)
                    if len(context_columns) > 0:
                        contextual_data_samples = [
                            _d
                            for _cc in context_columns
                            for _d in data_samples_list[table_name]
                            if _cc.lower() in _d.lower()
                        ]
                        data_samples_list = contextual_data_samples

                if len(context_columns) > 0:
                    filtered_dict = {k: f"{clmn_info_map[k]}" for k in context_columns}
                    filtered_c_type = ", ".join([f"{k} {v}" for k, v in filtered_dict.items()])
                _column_info = filtered_c_type if len(context_columns) > 0 else [columns_w_type]

                logger.debug(f"Relevant sample column values: {data_samples_list}")
                _table_name = ", ".join(table_names)

                query_prompt_format = STARCODER2_PROMPT
                if "h2ogpt-sql-nsql-llama-2-7B" in model_name:
                    query_prompt_format = NSQL_QUERY_PROMPT

                query = query_prompt_format.format(
                    table_name=_table_name,
                    column_info=_column_info,
                    data_info_detailed=data_samples_list,
                    sample_queries=qna_samples,
                    context=contextual_context_val,
                    question_txt=input_question,
                    dialect=self.dialect
                )

                logger.debug(f"Query Text:\n {query}")
                device_type = "cuda" if torch.cuda.is_available() else "cpu"

                # Check if the local models were selected
                current_temperature = 0.5
                tokenizer = model = generated_tokens = None
                if self.models and self.tokenizers and (model_name == "h2ogpt-sql-nsql-llama-2-7B-4bit" or model_name == "h2ogpt-sql-sqlcoder2-4bit" or model_name == "h2ogpt-sql-sqlcoder-34b-alpha-4bit"):
                    tokenizer = self.tokenizers[model_name]
                    inputs = tokenizer([query], return_tensors="pt")
                    model = self.models[model_name]
                    current_temperature = self.current_temps.get(model_name, 0.5)
                    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                    logger.info(f"Context length: {input_length}")

                    # Handle limited context length
                    # Currently, conservative approach: remove column description from the prompt, if input_length > (2048-300)
                    # Others to try:
                    # 1. Move to a model with larger context length
                    # 2. Possibly use a different tokenizer for chunking
                    # 3. Maybe positional interpolation --> https://arxiv.org/abs/2306.15595
                    if int(input_length) > 4000:
                        logger.info("Input length is greater than 1748, removing column description from the prompt")
                        query = query_prompt_format.format(
                            table_name=_table_name,
                            column_info=_column_info,
                            data_info_detailed="",
                            sample_queries=qna_samples,
                            context=contextual_context_val,
                            question_txt=input_question,
                        )
                        logger.debug(f"Adjusted query Text:\n {query}")
                        inputs = tokenizer([query], return_tensors="pt")
                        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                        logger.info(f"Adjusted context length: {input_length}")

                possible_temp_gt_5 = [0.6, 0.75, 0.8, 0.9, 1.0]
                possible_temp_lt_5 = [0.1, 0.2, 0.3, 0.4]
                random_seed = random.randint(0, 50)
                torch.manual_seed(random_seed)
                random_temperature = np.random.choice(possible_temp_lt_5, 1)[0] if current_temperature >= 0.5 else np.random.choice(possible_temp_gt_5, 1)[0]

                if not self.is_regenerate_with_options and not self.is_regenerate:
                    # Greedy decoding, for fast response
                    # Reset temperature to 0.5
                    current_temperature = 0.5
                    if model_name == "h2ogpt-sql-sqlcoder-7b-2" or model_name == "h2ogpt-sql-sqlcoder-34b-alpha" or model_name == "h2ogpt-sql-nsql-llama-2-7B":
                        m_name = MODEL_CHOICE_MAP_EVAL_MODE.get(model_name, "h2ogpt-sql-sqlcoder-34b-alpha")
                        query_txt = [{"role": "user", "content": query},]
                        logger.debug(f"Generation with default temperature : {current_temperature}")
                        completion = self.h2ogpt_client.with_options(max_retries=3).chat.completions.create(
                                    model=m_name,
                                    messages=query_txt,
                                    max_tokens=512,
                                    temperature=current_temperature,
                                    stop="```",
                                    seed=random_seed)
                        generated_tokens = completion.choices[0].message.content
                        logger.debug(f"Generated tokens: {generated_tokens}")
                    else:
                        if model:
                            model.eval()
                            output = model.generate(
                                **inputs.to(device_type),
                                max_new_tokens=512,
                                temperature=current_temperature,
                                output_scores=True,
                                do_sample=True,
                                return_dict_in_generate=True,
                            )

                            generated_tokens = output.sequences[:, input_length:][0]
                elif self.is_regenerate and not self.is_regenerate_with_options:
                    # throttle temperature for different result
                    logger.info("Regeneration requested on previous query ...")
                    logger.debug(f"Selected temperature for fast regeneration : {random_temperature}")
                    if model_name == "h2ogpt-sql-sqlcoder-7b-2" or model_name == "h2ogpt-sql-sqlcoder-34b-alpha" or model_name == "h2ogpt-sql-nsql-llama-2-7B":
                        m_name = MODEL_CHOICE_MAP_EVAL_MODE.get(model_name, "h2ogpt-sql-sqlcoder-34b-alpha")
                        query_txt = [{"role": "user", "content": query},]
                        completion = self.h2ogpt_client.with_options(max_retries=3).chat.completions.create(
                                    model=m_name,
                                    messages=query_txt,
                                    max_tokens=512,
                                    temperature=random_temperature,
                                    stop="```",
                                    seed=random_seed)
                        generated_tokens = completion.choices[0].message.content
                    else:
                        output = model.generate(
                            **inputs.to(device_type),
                            max_new_tokens=512,
                            temperature=random_temperature,
                            output_scores=True,
                            do_sample=True,
                            return_dict_in_generate=True,
                        )
                        generated_tokens = output.sequences[:, input_length:][0]
                    self.current_temps[model_name] = random_temperature
                    logger.debug(f"Temperature saved: {self.current_temps[model_name]}")
                else:
                    logger.info("Regeneration with options requested on previous query ...")
                    if model_name == "h2ogpt-sql-sqlcoder-7b-2" or model_name == "h2ogpt-sql-sqlcoder-34b-alpha" or model_name == "h2ogpt-sql-nsql-llama-2-7B":
                        logger.info("Generating diverse options, not enabled for remote models")
                        m_name = MODEL_CHOICE_MAP_EVAL_MODE.get(model_name, "h2ogpt-sql-sqlcoder-34b-alpha")
                        query_txt = [{"role": "user", "content": query},]
                        completion = self.h2ogpt_client.with_options(max_retries=3).chat.completions.create(
                                    model=m_name,
                                    messages=query_txt,
                                    max_tokens=512,
                                    temperature=random_temperature,
                                    stop="```",
                                    seed=random_seed)
                        generated_tokens = completion.choices[0].message.content
                    else:
                        # Diverse beam search decoding to explore more options
                        logger.debug(f"Selected temperature for diverse beam search: {random_temperature}")
                        output_re = model.generate(
                            **inputs.to(device_type),
                            max_new_tokens=512,
                            temperature=random_temperature,
                            top_k=5,
                            top_p=1.0,
                            num_beams=5,
                            num_beam_groups=5,
                            num_return_sequences=5,
                            output_scores=True,
                            do_sample=False,
                            diversity_penalty=2.0,
                            return_dict_in_generate=True,
                        )

                        transition_scores = model.compute_transition_scores(
                            output_re.sequences, output_re.scores, output_re.beam_indices, normalize_logits=False
                        )

                        # Create a boolean tensor where elements are True if the corresponding element in transition_scores is less than 0
                        mask = transition_scores < 0
                        # Sum the True values along axis 1
                        counts = torch.sum(mask, dim=1)
                        output_length = inputs.input_ids.shape[1] + counts
                        length_penalty = model.generation_config.length_penalty
                        reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)

                        # Converting logit scores to prob scores
                        probabilities_scores = F.softmax(reconstructed_scores, dim=-1)
                        out_idx = torch.argmax(probabilities_scores)
                        # Final output
                        output = output_re.sequences[out_idx]
                        generated_tokens = output[input_length:]

                        logger.info(f"Generated options:\n")
                        prob_sorted_idxs = sorted(
                            range(len(probabilities_scores)), key=lambda k: probabilities_scores[k], reverse=True
                        )
                        for idx, sorted_idx in enumerate(prob_sorted_idxs):
                            _out = output_re.sequences[sorted_idx]
                            res = tokenizer.decode(_out[input_length:], skip_special_tokens=True)
                            result = res.replace("table_name", _table_name).replace("```", "").strip()
                            if result.endswith(";"):
                                result = result.replace(";", "")
                            if "LIMIT".lower() not in result.lower():
                                res = "SELECT " + result.strip() + " LIMIT 100;"
                            else:
                                res = "SELECT " + result.strip() + ";"

                            pretty_sql = sqlparse.format(res, reindent=True, keyword_case="upper")
                            syntax_highlight = f"""``` sql\n{pretty_sql}\n```\n\n"""
                            alt_res = (
                                f"Option {idx+1}: (_probability_: {probabilities_scores[sorted_idx]})\n{syntax_highlight}\n"
                            )
                            alternate_queries.append(alt_res)
                            logger.info(f"Alternate options:\n{alt_res}")

                _res = generated_tokens
                if not self.remote_model and tokenizer:
                    _res = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Below is a pre-caution in-case of an error in table name during generation
                # COLLATE NOCASE is used to ignore case sensitivity, this might be specific to sqlite
                _temp = _res.replace("table_name", table_name) if _res and _res != '' else None
                res = _temp
                if not _temp:
                    res = None
                else:
                    if _temp.endswith("```"):
                        _temp = _temp.replace("```", "")
                    _temp = _temp.split("\n```")[0].strip()
                    # TODO Below should not happen, will have to check why its getting generated as part of response.
                    # Not sure, if its a vllm or prompt issue.
                    _temp = _temp.replace("/[/INST]", "").replace("[INST]", "").replace("[/INST]", "").strip()
                    if not _temp.lower().startswith('SELECT'.lower()):
                            _temp = "SELECT " + _temp.strip()
                            res = _temp
                    if "LIMIT".lower() not in _temp.lower():
                            _temp = _temp.strip().replace(";", "") + " LIMIT 100;"
                            res = _temp
                    else:
                        res = _temp.strip() + ";"

                # Validate the generate SQL for parsing errors, along with dialect specific validation
                # Note: Doesn't do well with handling date-time conversions
                # e.g.
                # sqlite: SELECT DATETIME(MAX(timestamp), '-5 minute') FROM demo WHERE isin_id = 'VM123'
                # postgres: SELECT MAX(timestamp) - INTERVAL '5 minutes' FROM demo where isin_id='VM123'
                # Reference ticket: https://github.com/tobymao/sqlglot/issues/2011
                result = res
                try:
                    result = sqlglot.transpile(res, identify=True, write=self.dialect)[0] if res else None
                except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
                    _, ex_value, ex_traceback = sys.exc_info()
                    logger.info(f"Attempting to fix syntax error ...,\n {e}")

                    h2o_client_url = os.getenv("H2OGPT_API_TOKEN", None)
                    h2o_client_key = os.getenv("H2OGPTE_API_TOKEN", None)
                    try:
                        result =  self.self_correction(input_query=res, error_msg=str(ex_traceback), remote_url=h2o_client_url, client_key=h2o_client_key)
                    except Exception as se:
                    # Another exception occurred, return the original SQL
                        logger.info(f"We did the best we could to fix syntactical error, there might be still be some issues:\n {se}")
                        logger.info(f"Realized query so far:\n {res}")
                        result = res
        return result, alternate_queries

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
