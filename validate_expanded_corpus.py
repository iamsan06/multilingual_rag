
from pathlib import Path
import pandas as pd

LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr",
         "ne", "or", "pa", "ta", "te", "ur"]

for lang in LANGS:
    benchmark_path = Path(f"data/raw/{lang}.parquet")
    distractor_path = Path(f"data/distractors/{lang}_distractors.parquet")

    benchmark = pd.read_parquet(benchmark_path)
    distractors = pd.read_parquet(distractor_path)

    # Normalize text consistently before checking overlap.
    gold_passages = set(
        benchmark["passage"].dropna().astype(str).str.strip()
    )
    distractor_passages = set(
        distractors["passage"].dropna().astype(str).str.strip()
    )

    overlap = gold_passages & distractor_passages
    combined = gold_passages | distractor_passages

    print(
        f"{lang}: queries={benchmark['query_id'].nunique()}, "
        f"gold_passages={len(gold_passages)}, "
        f"distractors={len(distractor_passages)}, "
        f"overlap={len(overlap)}, "
        f"combined_unique={len(combined)}"
    )
