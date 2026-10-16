# build_expanded_benchmark.py

import pandas as pd

from common import (
    LANGS,
    DATA_PROC,
    ROOT,
    load_benchmark,
    norm_text,
    text_id,
)

DISTRACTOR_DIR = ROOT / "data" / "distractors"
EXPECTED_DISTRACTORS = 10_000

for lang in LANGS:
    print(f"\n[{lang}] Building expanded corpus...")

    # 1. Load the existing benchmark and gold indices.
    queries, original_corpus, original_gold = load_benchmark(lang)

    # 2. Load the distractors already extracted.
    path = DISTRACTOR_DIR / f"{lang}_distractors.parquet"
    distractors = pd.read_parquet(path)

    distractor_texts = (
        distractors["passage"]
        .dropna()
        .map(norm_text)
    )
    distractor_texts = distractor_texts[distractor_texts != ""]

    # Do not silently evaluate an incomplete or duplicated distractor pool.
    assert len(distractor_texts) == EXPECTED_DISTRACTORS, (
        f"{lang}: expected 10,000 non-empty distractors, "
        f"found {len(distractor_texts)}"
    )
    assert distractor_texts.nunique() == EXPECTED_DISTRACTORS, (
        f"{lang}: distractors contain duplicates after normalization"
    )

    # 3. Normalize the original passages using the same convention.
    original_texts = original_corpus["text"].map(norm_text)

    # 4. Ensure distractors do not duplicate original passages.
    overlap = set(original_texts) & set(distractor_texts)
    assert not overlap, (
        f"{lang}: found {len(overlap)} distractor/original overlaps"
    )

    # 5. Combine original passages and distractors.
    combined_texts = list(original_texts) + list(distractor_texts)

    expanded_corpus = pd.DataFrame({
        "text": combined_texts,
    })
    expanded_corpus["doc_id"] = expanded_corpus["text"].map(text_id)

    # Final defensive deduplication and ID validation.
    expanded_corpus = expanded_corpus.drop_duplicates(
        subset="doc_id"
    ).reset_index(drop=True)

    assert expanded_corpus["doc_id"].is_unique
    assert len(expanded_corpus) == len(original_corpus) + EXPECTED_DISTRACTORS

    # 6. Map each query's gold document into the expanded corpus.
    doc_to_idx = {
        doc_id: idx
        for idx, doc_id in enumerate(expanded_corpus["doc_id"])
    }

    gold_doc_ids = queries["gold_doc_id"].tolist()
    expanded_gold = pd.Series(gold_doc_ids).map(doc_to_idx)

    assert expanded_gold.notna().all(), (
        f"{lang}: one or more gold passages are missing"
    )

    expanded_gold = expanded_gold.to_numpy(dtype="int64")

    # 7. Save the expanded corpus and its gold-index mapping.
    out_dir = DATA_PROC / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_corpus.to_parquet(
        out_dir / "corpus_expanded.parquet",
        index=False,
    )

    queries.to_parquet(
        out_dir / "queries_expanded.parquet",
        index=False,
    )

    pd.DataFrame({
        "query_id": queries["query_id"],
        "gold_doc_id": gold_doc_ids,
        "gold_idx": expanded_gold,
    }).to_parquet(
        out_dir / "gold_expanded.parquet",
        index=False,
    )

    print(
        f"{lang}: queries={len(queries)}, "
        f"original_docs={len(original_corpus)}, "
        f"distractors={len(distractor_texts)}, "
        f"expanded_docs={len(expanded_corpus)}"
    )

print("\nAll expanded benchmarks built successfully.")