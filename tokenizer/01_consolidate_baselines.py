# Tokenizer research — expanded-corpus baseline consolidation
# Source of truth: results/metrics.csv; never mix different corpus settings.

import pandas as pd
from pathlib import Path

METRICS = Path("results/metrics.csv")
OUT = Path("results")
LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur"]
COLUMNS = ["BM25", "mE5-small", "mE5-base", "mE5-large", "BGE-M3"]

df = pd.read_csv(METRICS)

required = {"setting", "model", "key", "MRR", "n_queries", "n_docs"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"metrics.csv is missing columns: {sorted(missing)}")

def build_table(setting, model_map, output_path):
    sub = df.loc[df["setting"].eq(setting)].copy()
    sub = sub.loc[sub["model"].isin(model_map)].copy()
    sub["model_label"] = sub["model"].map(model_map)

    dup = sub.duplicated(["key", "model_label"], keep=False)
    if dup.any():
        raise ValueError(
            "Duplicate rows found:\n"
            + sub.loc[dup, ["key", "model", "setting"]].to_string(index=False)
        )

    table = sub.pivot(index="key", columns="model_label", values="MRR").reindex(LANGS)
    table = table.reindex(columns=COLUMNS)
    table.index.name = "language"
    table.to_csv(output_path, float_format="%.6f")

    print(f"\nSaved {output_path}")
    print(table.round(4).to_string())
    print("\nMacro MRR (mean across available languages):")
    print(table.mean(skipna=True).round(4).to_string())
    print("\nCoverage (languages present):")
    print(sub.groupby("model_label")["key"].nunique().to_string())
    return table

# Model tags currently recorded for the original ~1k-passage corpus.
mono_models = {
    "bm25": "BM25",
    "mE5-small": "mE5-small",
    "mE5-base": "mE5-base",
    "mE5-large": "mE5-large",
    "bge-m3": "BGE-M3",
}

# Tags used by expanded-corpus runners. Add a tag here only after verifying it
# is the correct run on the expanded benchmark, not the original benchmark.
expanded_models = {
    "bm25_expanded": "BM25",
    "mE5-small-expanded": "mE5-small",
    "mE5-base-expanded": "mE5-base",
    "mE5-large-expanded": "mE5-large",
    "bge-m3-expanded": "BGE-M3",
}

build_table("mono", mono_models, OUT / "baseline_mrr_small.csv")
build_table("expanded_indicrag_10k", expanded_models, OUT / "baseline_mrr_expanded.csv")

coverage = (
    df.groupby(["setting", "model"])
      .agg(
          n_languages=("key", "nunique"),
          languages=("key", lambda values: ",".join(sorted(set(values)))),
          macro_mrr=("MRR", "mean"),
      )
      .reset_index()
)
coverage.to_csv(OUT / "baseline_coverage.csv", index=False)
print(f"\nSaved {OUT / 'baseline_coverage.csv'}")
print(coverage.to_string(index=False))
