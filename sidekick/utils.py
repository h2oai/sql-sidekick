import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sidekick.logger import logger
from sklearn.metrics.pairwise import cosine_similarity


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


def filter_samples(input_q: str, probable_qs: list, model_path: str, threshold: float = 0.45):
    # Only consider the questions, note: this might change in future.
    _inq = ("# query: " + input_q).strip().lower()
    logger.debug(f"Input questions: {_inq}")
    question_embeddings = generate_sentence_embeddings(model_path, x=[_inq], device="cpu")

    input_xs = [_se.split("# answer")[0].strip().lower() for _se in probable_qs]
    logger.debug(f"Probable questions: {input_xs}")
    embeddings = generate_sentence_embeddings(model_path, x=input_xs, device="cpu")
    res = []
    for idx, _se in enumerate(embeddings):
        similarities_score = cosine_similarity(
            [_se.astype(float).tolist()], [question_embeddings.astype(float).tolist()[0]]
        )
        logger.debug(f"Similarity score for: {input_xs[idx]}: {similarities_score[0][0]}")
        if similarities_score[0][0] > threshold:
            res.append(probable_qs[idx])
    return res


def remove_duplicates(input_x: list, model_path: str, threshold: float = 0.89):
    # Remove duplicates pairs
    embeddings = generate_sentence_embeddings(model_path, x=input_x, device="cpu")
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
    dir_list = [
        "var/lib/tmp/data",
        "var/lib/tmp/.cache",
        "var/lib/tmp/.cache/models/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2",
    ]
    for _dl in dir_list:
        p = Path(f"{base_path}/{_dl}")
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
