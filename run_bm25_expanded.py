# run_bm25_expanded.py

import time
import numpy as np
import pandas as pd

from common import (
    LANGS,
    DATA_PROC,
    BM25,
    tokenize,
    record,
)

# Start with Hindi only. Expand to all languages after it passes.
TEST_LANGS = LANGS

for lang in TEST_LANGS:
    print(f"\n[{lang}] Running BM25 on expanded corpus...")

    data_dir = DATA_PROC / lang

    queries = pd.read_parquet(
        data_dir / "queries_expanded.parquet"
    )
    corpus = pd.read_parquet(
        data_dir / "corpus_expanded.parquet"
    )
    gold_df = pd.read_parquet(
        data_dir / "gold_expanded.parquet"
    )

    gold = gold_df["gold_idx"].to_numpy(dtype=np.int64)

    # Verify the query and gold-label alignment.
    assert len(queries) == len(gold)
    assert np.all(gold >= 0)
    assert np.all(gold < len(corpus))

    print("Queries:", len(queries))
    print("Candidate passages:", len(corpus))

    start = time.time()

    # Tokenize every candidate passage and query.
    doc_tokens = [tokenize(t) for t in corpus["text"]]
    query_tokens = [tokenize(q) for q in queries["query"]]

    # Fit BM25 on the expanded corpus.
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(doc_tokens)

    # Score every query against every candidate passage.
    scores = bm25.score(query_tokens)

    assert scores.shape == (len(queries), len(corpus))

    # Save scores, ranks, and metrics separately from original BM25.
    metrics = record(
        "bm25_expanded",
        lang,
        scores,
        gold,
        setting="expanded_indicrag_10k",
    )

    print("Metrics:", metrics)
    print(f"Elapsed: {(time.time() - start):.1f} seconds")