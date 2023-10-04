import glob
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from InstructorEmbedding import INSTRUCTOR
from pandasql import sqldf
from sentence_transformers import SentenceTransformer
from logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def generate_sentence_embeddings(model_path: str, x, batch_size: int = 32, device: Optional[str] = None):
    # Reference:
    # 1. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    # 2. Evaluation result: https://www.sbert.net/_static/html/models_en_sentence_embeddings.html
    # 3. Model Card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # 4. Reference: https://huggingface.co/spaces/mteb/leaderboard
    # Maps sentence & paragraphs to a 384 dimensional dense vector space.
    model_name_path = f"{model_path}/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2/"
    current_torch_home = os.environ.get("TORCH_HOME", "")
    if Path(model_name_path).is_dir():
        is_empty = not any(Path(model_name_path).iterdir())
        if is_empty:
            # Download n cache at the specified location
            # https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip
            os.environ["TORCH_HOME"] = model_path
            model_name_path = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_model = SentenceTransformer(model_name_path, device=device)
    all_res = np.zeros(shape=(len(x), 0))
    res = sentence_model.encode(x, batch_size=batch_size, show_progress_bar=True)
    all_res = np.hstack((all_res, res))
    del sentence_model
    os.environ["TORCH_HOME"] = current_torch_home
    return all_res


def load_embedding_model(model_path: str, device: str):
    model_name_path = glob.glob(f"{model_path}/models--BAAI--bge-base-en/snapshots/*/")[0]

    sentence_model = SentenceTransformer(model_name_path, cache_folder=model_path, device=device)
    if "cuda" not in device:
        # Issue https://github.com/pytorch/pytorch/issues/69364
        # # In the initial experimentation, quantized model is generates slightly better results
        logger.debug("Sentence embedding model is quantized ...")
        model_obj = torch.quantization.quantize_dynamic(sentence_model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        model_obj = sentence_model
    return model_obj


def generate_text_embeddings(model_path: str, x, model_obj=None, batch_size: int = 32, device: Optional[str] = "cpu"):
    # Reference:
    # 1. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    # Maps sentence & paragraphs to a 384 dimensional dense vector space.
    if model_obj is None:
        model_obj = load_embedding_model(model_path, device)

    _sentences = [["Represent the Financial question for retrieving duplicate examples: ", _item] for _item in x]

    res = model_obj.encode(_sentences)
    return res


def filter_samples(
    input_q: str,
    probable_qs: list,
    model_path: str,
    model_obj=None,
    threshold: float = 0.80,
    device="auto",
    is_regenerate: bool = False,
):
    # Only consider the questions, note: this might change in future.
    _inq = ("# query: " + input_q).strip().lower()
    logger.debug(f"Input questions: {_inq}")
    _device = "cuda" if torch.cuda.is_available() else "cpu" if device == "auto" else device
    question_embeddings = generate_text_embeddings(model_path, x=[_inq], model_obj=model_obj, device=_device)

    input_pqs = [_se.split("# answer")[0].strip().lower() for _se in probable_qs]
    logger.debug(f"Probable context: {input_pqs}")
    embeddings = generate_text_embeddings(model_path, x=input_pqs, model_obj=model_obj, device=_device)
    res = {}
    _scores = {}
    for idx, _se in enumerate(embeddings):
        similarities_score = cosine_similarity(
            [_se.astype(float).tolist()], [question_embeddings.astype(float).tolist()[0]]
        )
        logger.debug(f"Similarity score for: {input_pqs[idx]}: {similarities_score[0][0]}")
        _scores[idx] = similarities_score[0][0]
        if similarities_score[0][0] > threshold:
            res[str(probable_qs[idx])] = similarities_score[0][0]

    # Get Top N Context Queries if user requested to regenerate regardless of scores
    if len(res) == 0 and is_regenerate and len(_scores) > 0:
        top_n = min(len(_scores), 2)
        sorted_res = dict()
        sorted_scores = sorted(_scores, key=_scores.get, reverse=True)
        top_idxs = sorted_scores[:top_n]
        for idx in top_idxs:
            sorted_res[str(probable_qs[idx])] = similarities_score[0][0]
    else:
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)

    logger.debug(f"Sorted context: {sorted_res}")
    return list(dict(sorted_res).keys())


def remove_duplicates(
    input_x: list, model_path: str, similarity_model=None, threshold: float = 0.989, device: str = "cpu"
):
    # Remove duplicates pairs
    if input_x is None or len(input_x) < 2:
        res = []
    else:
        embeddings, _ = generate_text_embeddings(model_path, x=input_x, model_obj=similarity_model, device=device)
        similarity_scores = cosine_similarity(embeddings)
        similar_indices = [(x, y) for (x, y) in np.argwhere(similarity_scores > threshold) if x != y]

        # Remove identical pairs e.g. [(0, 3), (3, 0)] -> [(0, 3)]
        si = [similarity_scores[tpl] for tpl in similar_indices]
        dup_pairs_idx = np.where(pd.Series(si).duplicated())[0].tolist()
        remove_vals = [similar_indices[_itm] for _itm in dup_pairs_idx]
        [similar_indices.remove(_itm) for _itm in remove_vals]
        res = list(set([item[0] for item in similar_indices]))
    return res


def save_query(output_path: str, query, response, extracted_entity: Optional[dict] = ""):
    _response = response
    # Probably need to find a better way to extra the info rather than depending on key phrases
    if response and "Generated response for question,".lower() in response.lower():
        _response = response.split("**Generated response for question,**")[1].split("\n")[3].strip()
    chat_history = {"Query": query, "Answer": _response, "Entity": extracted_entity}

    with open(f"{output_path}/var/lib/tmp/data/history.jsonl", "a") as outfile:
        json.dump(chat_history, outfile)
        outfile.write("\n")


def setup_dir(base_path: str):
    dir_list = ["var/lib/tmp/data", "var/lib/tmp/jobs", "var/lib/tmp/.cache", "models/weights"]
    for _dl in dir_list:
        p = Path(f"{base_path}/{_dl}")
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)


def update_tables(json_file_path: str, new_data: dict):
    # Check if the JSON file exists
    if os.path.exists(json_file_path):
        try:
            # Read the existing content from the JSON file
            with open(json_file_path, "r") as json_file:
                existing_data = json.load(json_file)
            logger.debug("Existing Data:", existing_data)
        except Exception as e:
            logger.debug(f"An error occurred while reading: {e}")
    else:
        existing_data = {}
        logger.debug("JSON file doesn't exist. Creating a new one.")

    # Append new data to the existing content
    existing_data.update(new_data)

    # Write the updated content back to the JSON file
    try:
        with open(json_file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)
        logger.debug("Data appended and file updated.")
    except Exception as e:
        logger.debug(f"An error occurred while writing: {e}")


def read_sample_pairs(input_path: str, model_name: str = "h2ogpt-sql"):
    df = pd.read_csv(input_path)
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # NSQL format
    if model_name != "h2ogpt-sql":
        # Open AI format
        # Convert frame to below format
        # [
        # "# query": ""
        # "# answer": ""
        # ]
        res = df.apply(lambda row: f"# query: {row['query']}\n# answer: {row['answer']}", axis=1).to_list()
    else:
        # Convert frame to below format
        # [
        # "Question": <question_text>
        # "Answer":
        # <response_text>
        # ]
        res = df.apply(lambda row: f"Question: {row['query']}\nAnswer:\n{row['answer']}", axis=1).to_list()
    return res


def extract_table_names(query: str):
    """
    Extracts table names from a SQL query.

    Parameters:
        query (str): The SQL query to extract table names from.

    Returns:
        list: A list of table names.
    """
    table_names = re.findall(r"\bFROM\s+(\w+)", query, re.IGNORECASE)
    table_names += re.findall(r"\bJOIN\s+(\w+)", query, re.IGNORECASE)
    table_names += re.findall(r"\bUPDATE\s+(\w+)", query, re.IGNORECASE)
    table_names += re.findall(r"\bINTO\s+(\w+)", query, re.IGNORECASE)

    # Below keywords may not be relevant for the project but adding for sake for completeness
    table_names += re.findall(r"\bINSERT\s+INTO\s+(\w+)", query, re.IGNORECASE)
    table_names += re.findall(r"\bDELETE\s+FROM\s+(\w+)", query, re.IGNORECASE)
    return np.unique(table_names).tolist()


def execute_query_pd(query=None, tables_path=None, n_rows=100):
    """
    Runs an SQL query on a pandas DataFrame.

    Parameters:
        df (pandas DataFrame): The DataFrame to query.
        query (str): The SQL query to execute.

    Returns:
        pandas DataFrame: The result of the SQL query.
    """
    for table in tables_path:
        if not table in locals():
            # Update the local namespace with the table name, pandas object
            locals()[f"{table}"] = pd.read_csv(tables_path[table])

    res_df = sqldf(query, locals())
    return res_df


def get_table_keys(file_path: str, table_key: str):
    res = []
    if not os.path.exists(file_path):
        logger.debug(f"File '{file_path}' does not exist.")
        return res, dict()

    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        if isinstance(data, dict):
            res = list(data.keys())
    if table_key:
        return None, data[table_key]
    else:
        return res, data


def is_resource_low():
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024**3)
    logger.info(f"Total Memory: {total_memory}GB")
    logger.info(f"Free GPU memory: {free_in_GB}GB")
    off_load = True
    if (int(free_in_GB) - 2) >= int(0.5 * total_memory):
        off_load = False
    return off_load


def load_causal_lm_model(
    model_name: str,
    cache_path: str,
    device: str,
    load_in_8bit: bool = False,
    load_in_4bit=True,
    off_load: bool = False,
    re_generate: bool = False,
):
    try:
        # Load h2oGPT.NSQL model
        device = {"": 0} if torch.cuda.is_available() else "cpu" if device == "auto" else device
        total_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024**3)
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
        logger.info(f"Free GPU memory: {free_in_GB}GB")
        n_gpus = torch.cuda.device_count()
        _load_in_8bit = load_in_8bit

        # 22GB (Least requirement on GPU) is a magic number for the current model size.
        if off_load and re_generate and total_memory < 22:
            # To prevent the system from crashing in-case memory runs low.
            # TODO: Performance when offloading to CPU.
            max_memory = f"{4}GB"
            max_memory = {i: max_memory for i in range(n_gpus)}
            logger.info(f"Max Memory: {max_memory}, offloading to CPU")
            with init_empty_weights():
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_path, offload_folder=cache_path)
                # A blank model with desired config.
                model = AutoModelForCausalLM.from_config(config)
                device = infer_auto_device_map(model, max_memory=max_memory)
                device["lm_head"] = 0
            _offload_state_dict = True
            _llm_int8_enable_fp32_cpu_offload = True
            _load_in_8bit = True
            load_in_4bit = False
        else:
            max_memory = f"{int(free_in_GB)-2}GB"
            max_memory = {i: max_memory for i in range(n_gpus)}
            _offload_state_dict = False
            _llm_int8_enable_fp32_cpu_offload = False

        if _load_in_8bit and _offload_state_dict and not load_in_4bit:
            _load_in_8bit = False if "cpu" in device else True
            logger.debug(
                f"Loading in 8 bit mode: {_load_in_8bit} with offloading state: {_llm_int8_enable_fp32_cpu_offload}"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_path,
                device_map=device,
                load_in_8bit=_load_in_8bit,
                llm_int8_enable_fp32_cpu_offload=_llm_int8_enable_fp32_cpu_offload,
                offload_state_dict=_offload_state_dict,
                max_memory=max_memory,
                offload_folder=f"{cache_path}/weights/",
            )

        else:
            logger.debug(f"Loading in 4 bit mode: {load_in_4bit} with device {device}")
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=cache_path, device_map=device, quantization_config=nf4_config
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path, device_map=device, use_fast=True)

        return model, tokenizer
    except Exception as e:
        logger.info(f"An error occurred while loading the model: {e}")
        return None, None


def _check_file_info(file_path: str):
    if file_path is not None and Path(file_path).exists():
        logger.info(f"Using information info from path {file_path}")
        return file_path
    else:
        logger.info("Required info not found, provide a path for table information and try again")
        raise FileNotFoundError(f"Table info not found at {file_path}")


def _execute_sql(query: str):
    # Check for,
    # 1. Keyword: "Execute SQL: <SQL query>"
    # 2. Query starts with SQL statement
    # TODO vulnerability check for possible SELECT SQL injection via source code.
    _cond1 = _cond2 = False
    _cond1 = re.findall(r"Execute SQL:\s+(.*)", query, re.IGNORECASE)
    _temp_cond = query.strip().lower().split("execute sql:")
    if len(_temp_cond) > 1:
        _cond2 = True if query.strip().lower().split("execute sql:")[1].strip().startswith("select") else False
    return _cond1 and _cond2


def make_dir(path: str):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EXIST and os.path.isdir(path):
            pass
        else:
            raise Exception("Error reported while creating default directory path.")


def flatten_list(_list: list):
    return [item for sublist in _list for item in sublist]


def check_vulnerability(input_query: str):
    # Common SQL injection patterns checklist
    # Reference: https://github.com/payloadbox/sql-injection-payload-list#generic-sql-injection-payloads
    sql_injection_patterns = [
        r"\b(UNION\s+ALL\s+SELECT|OR\s+\d+\s*=\s*\d+|1\s*=\s*1|--\s+)",
        r'\b(SELECT\s+\*\s+FROM\s+\w+\s+WHERE\s+\w+\s*=\s*[\'"].*?[\'"]\s*;?\s*--)',
        r'\b(INSERT\s+INTO\s+\w+\s+\(\s*\w+\s*,\s*\w+\s*\)\s+VALUES\s*\(\s*[\'"].*?[\'"]\s*,\s*[\'"].*?[\'"]\s*\)\s*;?\s*--)',
        r"\b(DROP\s+TABLE|ALTER\s+TABLE|admin\'--)",  # DROP TABLE/ALTER TABLE
        r"(?:'|\”|--|#|‘\s*OR\s*‘1|‘\s*OR\s*\d+\s*--\s*-|\"\s*OR\s*\"\" = \"|\"\s*OR\s*\d+\s*=\s*\d+\s*--\s*-|’\s*OR\s*''\s*=\s*‘|‘=‘|'=0--+|OR\s*\d+\s*=\s*\d+|‘\s*OR\s*‘x’=‘x’|AND\s*id\s*IS\s*NULL;\s*--|‘’’’’’’’’’’’’UNION\s*SELECT\s*‘\d+|%00|/\*.*?\*/|\+|\|\||%|@\w+|@@\w+)",
        r"AND\s[01]|AND\s(true|false)|[01]-((true|false))",
        r"\d+'\s*ORDER\s*BY\s*\d+--\+|\d+'\s*GROUP\s*BY\s*(\d+,)*\d+--\+|'\s*GROUP\s*BY\s*columnnames\s*having\s*1=1\s*--",
        r"\bUNION\b\s+\b(?:ALL\s+)?\bSELECT\b\s+[A-Za-z0-9]+",  # Union Based
        r'\b(OR|AND|HAVING|AS|WHERE)\s+\d+=\d+(\s+AND\s+[\'"]\w+[\'"]\s*=\s*[\'"]\w+[\'"])?(\s*--|\s*#)?\b',
        r"\b(?:RLIKE|IF)\s*\(\s*SELECT\s*\(\s*CASE\s*WHEN\s*\(\s*[\d=]+\s*\)\s*THEN\s*0x[0-9a-fA-F]+\s*ELSE\s*0x[0-9a-fA-F]+\s*END\s*\)\s*\)\s*AND\s*'\w+'=\w+\s*|\b%\s*AND\s*[\d=]+\s*AND\s*'\w+'=\w+\s*|and\s*\(\s*select\s*substring\s*\(\s*@@version,\d+,\d+\)\s*\)=\s*'[\w]'\b",
        r"('|\")?\s*(or|\|\|)\s*sleep\(.*?\)\s*(\#|--)?\s*(;waitfor\s+delay\s+'[0-9:]+')?\s*;?(\s+AND\s+)?\s*\w+\s*=\s*\w+\s*",  # Time Based
        r"(ORDER BY \d+,\s*)*(ORDER BY \d+,?)*SLEEP\(\d+\),?(BENCHMARK\(\d+,\s*MD5\('[A-Z]'\)\),?)*\d*,?",  # Additional generic UNION patterns
    ]

    # Check for SQL injection patterns in the SQL code
    res = False
    _msg = None
    p_detected = []
    for pattern in sql_injection_patterns:
        matches = re.findall(pattern, input_query, re.IGNORECASE)
        if matches:
            if all(v == "'" for v in matches) or all(v == "''" for v in matches):
                matches = []
            else:
                res = True
                p_detected.append(matches)
    _pd = set(flatten_list(p_detected))
    if res:
        _detected_patterns = ", ".join([str(elem) for elem in _pd])
        _msg = f"The input question has malicious patterns, **{_detected_patterns}** that could lead to SQL Injection.\nSorry, I will not be able to provide an answer.\nPlease try rephrasing the question."
    return res, _msg
