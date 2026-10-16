# run_bge_expanded.py

import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import LANGS, DATA_PROC, RUN_DIR, RES_DIR, record

MODEL_NAME = "BAAI/bge-m3"
MODEL_TAG = "bge-m3-expanded"
SETTING = "expanded_indicrag_10k"

BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    torch.set_num_threads(12)

RUN_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = DATA_PROC.parent.parent / "embeddings" / MODEL_TAG
EMB_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Model:", MODEL_NAME)
print("Embedding cache:", EMB_DIR)
print("Languages:", len(LANGS))

model = SentenceTransformer(MODEL_NAME, device=DEVICE)
model.max_seq_length = MAX_SEQ_LENGTH

print("Embedding dimension:", model.get_sentence_embedding_dimension())

assert model.get_sentence_embedding_dimension() == 1024

# --------------------------------------------------
# 3. Cache helper with ordering validation
# --------------------------------------------------
def fingerprint(texts):
    h = hashlib.sha256()
    for text in texts:
        h.update(str(text).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def encode_cached(texts, cache_path, expected_fingerprint):
    meta_path = cache_path.with_suffix(".json")

    if cache_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cached = np.load(cache_path)

        if (
            meta.get("fingerprint") == expected_fingerprint
            and cached.shape == (len(texts), 1024)
        ):
            print("Using cache:", cache_path.name)
            return cached.astype(np.float32, copy=False)

        print("Cache mismatch; re-encoding:", cache_path.name)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    assert embeddings.shape == (len(texts), 1024)
    assert np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4
    )

    np.save(cache_path, embeddings)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fingerprint": expected_fingerprint,
                "shape": list(embeddings.shape),
                "model": MODEL_NAME,
            },
            f,
            indent=2,
        )

    return embeddings


# --------------------------------------------------
# 4. Evaluate each expanded language
# --------------------------------------------------
failed = []

for lang in LANGS:
    print("\n" + "=" * 65)
    print("Language:", lang)

    score_path = RUN_DIR / MODEL_TAG / f"{lang}_scores.npy"

    # Avoid accidentally reusing original-corpus BGE scores.
    if score_path.exists():
        print("Scores already exist; skipping:", score_path)
        continue

    try:
        start = time.time()

        data_dir = DATA_PROC / lang

        queries = pd.read_parquet(data_dir / "queries_expanded.parquet")
        corpus = pd.read_parquet(data_dir / "corpus_expanded.parquet")
        gold_df = pd.read_parquet(data_dir / "gold_expanded.parquet")

        query_texts = queries["query"].astype(str).tolist()
        documents = corpus["text"].astype(str).tolist()
        gold = gold_df["gold_idx"].to_numpy(dtype=np.int64)

        # Verify alignment and gold mappings.
        assert len(query_texts) == len(gold)
        assert len(documents) in (10999, 11000)
        assert np.all((gold >= 0) & (gold < len(documents)))

        assert (
            corpus["doc_id"].iloc[gold].to_numpy()
            == queries["gold_doc_id"].to_numpy()
        ).all(), f"{lang}: gold passage mapping mismatch"

        print("Queries:", len(query_texts))
        print("Candidate passages:", len(documents))

        q_fp = fingerprint(query_texts)
        d_fp = fingerprint(corpus["doc_id"].astype(str).tolist())

        # Encode queries and documents in their fixed canonical order.
        Eq = encode_cached(
            query_texts,
            EMB_DIR / f"{lang}_queries.npy",
            q_fp,
        )

        Ed = encode_cached(
            documents,
            EMB_DIR / f"{lang}_documents.npy",
            d_fp,
        )

        # L2-normalized embeddings -> dot product equals cosine similarity.
        scores = Eq @ Ed.T

        assert scores.shape == (len(query_texts), len(documents))

        metrics = record(
            model=MODEL_TAG,
            key=lang,
            S=scores,
            gold=gold,
            setting=SETTING,
        )

        print(
            f"{lang} | MRR={metrics['MRR']:.4f} | "
            f"R@10={metrics['R@10']:.4f} | "
            f"R@100={metrics['R@100']:.4f} | "
            f"Time={(time.time() - start) / 60:.2f} min"
        )

        del queries, corpus, gold_df, Eq, Ed, scores
        gc.collect()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"FAILED: {lang}: {repr(e)}")
        failed.append(lang)

print("\n" + "=" * 65)
print("BGE-M3 EXPANDED RUN FINISHED")
print("Failed languages:", failed if failed else "none")

del model
gc.collect()

metrics_df = pd.read_csv(RES_DIR / "metrics.csv")

bge_results = metrics_df[
    (metrics_df["model"] == MODEL_TAG)
    & (metrics_df["setting"] == SETTING)
].copy()

print("\nCompleted languages:", bge_results["key"].nunique())
print(
    bge_results[
        ["key", "n_docs", "n_queries", "MRR", "MRR@10", "R@10"]
    ].sort_values("key").to_string(index=False)
)