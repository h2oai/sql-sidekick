from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_sentence_embeddings(model_path: str, x, batch_size: int = 32, device: Optional[str] = None):
    # Reference:
    # 1. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    # 2. Evaluation result: https://www.sbert.net/_static/html/models_en_sentence_embeddings.html
    # 3. Model Card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    # 4. Reference: https://huggingface.co/spaces/mteb/leaderboard
    # Maps sentence & paragraphs to a 384 dimensional dense vector space.
    sentence_model = SentenceTransformer(model_path, device=device)
    all_res = np.zeros(shape=(len(x), 0))
    res = sentence_model.encode(x, batch_size=batch_size, show_progress_bar=True)
    all_res = np.hstack((all_res, res))
    del sentence_model
    return all_res


def compute_similarity(vectors: list):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(vectors)
    return similarity


def remove_duplicates(input_x: list, model_path: str, threshold: float = 0.89):
    # Remove duplicates pairs
    embeddings = generate_sentence_embeddings(model_path, x=input_x, device="cpu")
    similarity_scores = compute_similarity(embeddings)
    similar_indices = [(x, y) for (x, y) in np.argwhere(similarity_scores > threshold) if x != y]

    # Remove identical pairs e.g. [(0, 3), (3, 0)] -> [(0, 3)]
    si = [similarity_scores[tpl] for tpl in similar_indices]
    dup_pairs_idx = np.where(pd.Series(si).duplicated())[0].tolist()
    remove_vals = [similar_indices[_itm] for _itm in dup_pairs_idx]
    [similar_indices.remove(_itm) for _itm in remove_vals]
    res = list(set([item[0] for item in similar_indices]))
    return res
