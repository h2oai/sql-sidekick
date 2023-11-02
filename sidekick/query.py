import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import openai
import sqlglot
import sqlparse
import torch
import torch.nn.functional as F
from langchain import OpenAI
from llama_index import GPTSimpleVectorIndex, GPTSQLStructStoreIndex, LLMPredictor, ServiceContext, SQLDatabase
from llama_index.indices.struct_store import SQLContextContainerBuilder
from sidekick.configs.prompt_template import (
    DEBUGGING_PROMPT,
    NSQL_QUERY_PROMPT,
    QUERY_PROMPT,
    STARCODER2_PROMPT,
    TASK_PROMPT,
)
from sidekick.logger import logger
from sidekick.utils import (
    _check_file_info,
    is_resource_low,
    load_causal_lm_model,
    load_embedding_model,
    make_dir,
    re_rank,
    read_sample_pairs,
    remove_duplicates,
    semantic_search,
)
from sqlalchemy import create_engine


class SQLGenerator:
    _instance = None

    def __new__(
        cls,
        db_url: str,
        openai_key: str = None,
        model_name="h2ogpt-sql-nsql-llama-2-7B",
        data_input_path: str = "./table_info.jsonl",
        sample_queries_path: str = "./samples.csv",
        job_path: str = "./",
        device: str = "auto",
        is_regenerate: bool = False,
        is_regenerate_with_options: bool = False,
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
            if cls._instance.models.get(cls._instance.model_name, None):
                _name = cls._instance.model_name
                del cls._instance.models[_name]
                cls._instance.models[_name] = None

            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Low memory: {offloading}/ Model re-initialization: {is_regenerate_with_options}")

        if cls._instance is None or (cls._instance and not cls._instance.models.get(model_name, None)):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            cls._instance.models, cls._instance.tokenizers = load_causal_lm_model(
                model_name,
                cache_path=f"{job_path}/models/",
                device=device,
                off_load=offloading,
                re_generate=is_regenerate_with_options,
            )
            cls._instance.model_name = "h2ogpt-sql-sqlcoder2" if not model_name else model_name
            model_embed_path = f"{job_path}/models/sentence_transformers"
            cls._instance.models[cls._instance.model_name].current_temperature = 0.5
            device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
            cls._instance.similarity_model = load_embedding_model(model_path=model_embed_path, device=device)
        return cls._instance

    def __init__(
        self,
        db_url: str,
        openai_key: str = None,
        model_name="h2ogpt-sql-nsql-llama-2-7B",
        data_input_path: str = "./table_info.jsonl",
        sample_queries_path: str = "./samples.csv",
        job_path: str = "./",
        device: str = "cpu",
        is_regenerate: bool = False,
        is_regenerate_with_options: bool = False,
    ):
        self.db_url = db_url
        self.engine = create_engine(db_url) if db_url else None
        self.sql_database = SQLDatabase(self.engine) if self.engine else None
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
        self.table_name = None

    def clear(self):
        del SQLGenerator._instance
        SQLGenerator._instance = None

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
        _dialect: str = "sqlite",
        model_name: str = "h2ogpt-sql-nsql-llama-2-7B",
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

            if "h2ogpt-sql" not in model_name:
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
                service_context_gpt3 = ServiceContext.from_defaults(
                    llm_predictor=llm_predictor_gpt3, chunk_size_limit=512
                )

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
                    res = sqlglot.transpile(res, read=_dialect)
                except (sqlglot.errors.ParseError, ValueError, RuntimeError) as e:
                    logger.info("We did the best we could, there might be still be some error:\n")
                    logger.info(f"Realized query so far:\n {res}")
            else:
                # TODO Update needed for multiple tables
                columns_w_type = (
                    self.context_builder.full_context_dict[table_name]
                    .split(":")[2]
                    .split(" and foreign keys")[0]
                    .strip()
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
                clmn_names = [i.split("(")[0].strip() for i in column_names]
                clmn_types = [i.split("(")[1].strip().replace(")", "") for i in column_names]
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
                    filtered_dict = {k: f"({clmn_info_map[k]})" for k in context_columns}
                    filtered_c_type = ", ".join([f"{k} {v}" for k, v in filtered_dict.items()])
                _column_info = filtered_c_type if len(context_columns) > 0 else [columns_w_type]

                logger.debug(f"Relevant sample column values: {data_samples_list}")
                _table_name = ", ".join(table_names)

                query_prompt_format = STARCODER2_PROMPT
                if model_name == "h2ogpt-sql-nsql-llama-2-7B":
                    query_prompt_format = NSQL_QUERY_PROMPT

                query = query_prompt_format.format(
                    table_name=_table_name,
                    column_info=_column_info,
                    data_info_detailed=data_samples_list,
                    sample_queries=qna_samples,
                    context=contextual_context_val,
                    question_txt=input_question,
                )

                logger.debug(f"Query Text:\n {query}")
                tokenizer = self.tokenizers[model_name]
                inputs = tokenizer([query], return_tensors="pt")
                model = self.models[model_name]
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
                # Generate SQL
                random_seed = random.randint(0, 50)
                torch.manual_seed(random_seed)

                # Greedy search for quick response
                model.eval()
                device_type = "cuda" if torch.cuda.is_available() else "cpu"

                possible_temp_gt_5 = [0.6, 0.75, 0.8, 0.9, 1.0]
                possible_temp_lt_5 = [0.1, 0.2, 0.3, 0.4]
                random_temperature = model.current_temperature
                random_seed = random.randint(0, 50)
                torch.manual_seed(random_seed)
                if model.current_temperature >= 0.5:
                    random_temperature = np.random.choice(possible_temp_lt_5, 1)[0]
                else:
                    random_temperature = np.random.choice(possible_temp_gt_5, 1)[0]
                if not self.is_regenerate_with_options and not self.is_regenerate:
                    # Greedy decoding
                    output = model.generate(
                        **inputs.to(device_type),
                        max_new_tokens=512,
                        temperature=0.5,
                        output_scores=True,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )

                    generated_tokens = output.sequences[:, input_length:][0]
                elif self.is_regenerate and not self.is_regenerate_with_options:
                    # throttle temperature for different result
                    logger.info("Regeneration requested on previous query ...")
                    logger.debug(f"Selected temperature for fast regeneration : {random_temperature}")
                    output = model.generate(
                        **inputs.to(device_type),
                        max_new_tokens=512,
                        temperature=random_temperature,
                        output_scores=True,
                        do_sample=True,
                        return_dict_in_generate=True,
                    )
                    generated_tokens = output.sequences[:, input_length:][0]
                    model.current_temperature = random_temperature
                else:
                    logger.info("Regeneration with options requested on previous query ...")
                    # Diverse beam search decoding to explore more options
                    logger.debug(f"Selected temperature for diverse beam search: {random_temperature}")
                    output_re = model.generate(
                        **inputs.to(device_type),
                        max_new_tokens=512,
                        temperature=random_temperature,
                        top_k=5,
                        top_p=0.9,
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
                        result = res.replace("table_name", _table_name)
                        # Remove the last semi-colon if exists at the end
                        # we will add it later
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
                        logger.info(alt_res)

                _res = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Below is a pre-caution in-case of an error in table name during generation
                # COLLATE NOCASE is used to ignore case sensitivity, this might be specific to sqlite
                _temp = _res.replace("table_name", table_name).split(";")[0]

                if _temp.endswith(";"):
                    _temp = _temp.replace(";", "")
                if "LIMIT".lower() not in _temp.lower():
                    res = "SELECT " + _temp.strip() + " LIMIT 100;"
                else:
                    res = "SELECT " + _temp.strip() + ";"

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
