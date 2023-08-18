import glob
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from InstructorEmbedding import INSTRUCTOR
from pandasql import sqldf
from sentence_transformers import SentenceTransformer
from sidekick.logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    model_name_path = glob.glob(f"{model_path}/models--hkunlp--instructor-large/snapshots/*/")[0]

    sentence_model = INSTRUCTOR(model_name_path, cache_folder=model_path, device=device)
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
    input_q: str, probable_qs: list, model_path: str, model_obj=None, threshold: float = 0.80, device="auto"
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
    _scores = []
    for idx, _se in enumerate(embeddings):
        similarities_score = cosine_similarity(
            [_se.astype(float).tolist()], [question_embeddings.astype(float).tolist()[0]]
        )
        logger.debug(f"Similarity score for: {input_pqs[idx]}: {similarities_score[0][0]}")
        if similarities_score[0][0] > threshold:
            res[str(probable_qs[idx])] = similarities_score[0][0]
            _scores.append(similarities_score[0][0])

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
    chat_history = {"Query": query, "Answer": response, "Entity": extracted_entity}

    with open(f"{output_path}/var/lib/tmp/data/history.jsonl", "a") as outfile:
        json.dump(chat_history, outfile)
        outfile.write("\n")


def setup_dir(base_path: str):
    dir_list = ["var/lib/tmp/data", "var/lib/tmp/jobs", "var/lib/tmp/.cache"]
    for _dl in dir_list:
        p = Path(f"{base_path}/{_dl}")
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)


def update_tables(json_file_path:str, new_data:dict):
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


def get_table_keys(file_path:str, table_key:str):
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


def load_causal_lm_model(model_name: str, cache_path: str, device: str, load_in_8bit: bool = True):
    try:
        # Load h2oGPT.NSQL model
        device = {"": 0} if torch.cuda.is_available() else "cpu" if device == "auto" else device
        if load_in_8bit:
            _load_in_8bit = False if "cpu" in device else True
        else:
            _load_in_8bit = False
        cache_path = f"{cache_path}/models/"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path, device_map=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_path, device_map=device, load_in_8bit=_load_in_8bit
        )

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


def make_dir(path: str):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EXIST and os.path.isdir(path):
            pass
        else:
            raise Exception("Error reported while creating default directory path.")
