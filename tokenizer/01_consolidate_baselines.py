# Tokenizer research — expanded-corpus baseline consolidation
# Source of truth: results/metrics.csv; do not mix settings.

import pandas as pd
from pathlib import Path

METRICS = Path("results/metrics.csv")
OUT = Path("results")

df = pd.read_csv(METRICS)

required = {"setting", "model", "key", "MRR", "MRR@10", "R@1", "R@5", "R@10", "R@20", "R@100", "nDCG@10", "n_queries", "n_docs"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"metrics.csv is missing columns: {sorted(missing)}")

# Small-corpus model tag names seen in the existing CSV.
mono_models = {
    "bm25": "BM25",
    "mE5-small": "mE5-small",
    "mE5-base": "mE5-base",
    "mE5-large": "mE5-large",
    "bge-m3": "BGE-M3",
}
expanded_models = {
    "bm25_expanded": "BM25",
    "mE5-small-expanded": "mE5-small",
    "mE5-base-expanded": "mE5-base",
    "mE5-large-expanded": "mE5-large",
    "bge-m3-expanded": "BGE-M3",
}

LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur"]

def build_table(setting, model_map, output_path):
    sub = df[df["setting"] == setting].copy()
    sub = sub[sub["model"].isin(model_map)].copy()
    sub["model_label"] = sub["model"].map(model_map)

    # Fail loudly on duplicate language/model results rather than silently picking one.
    dup = sub.duplicated(["key", "model_label"], keep=False)
    if dup.any():
        raise ValueError("Duplicate rows found:\n" + sub.loc[dup, ["key", "model", "setting"]].to_string(index=False))

    table = sub.pivot(index="key", columns="model_label", values="MRR").reindex(LANGS)
    # Retain columns even where an expanded dense baseline has not been run/saved.
    for name in ["BM25", "mE5-small", "mE5-base", "mE5-large", "BGE-M3"]:
        if name not in table.columns:
            table[name] = float("nan")
    table = table[["BM25", "mE5-small", "mE5-base", "mE5-large", "BGE-M3"]]
    table.index.name = "language"
    table.to_csv(output_path, float_format="%.6f")
    print(f"Saved {output_path}")
    print(table.round(4).to_string())
    print("\nMacro MRR (mean across available languages):")
    print(table.mean(skipna=True).round(4).to_string())

# Small corpus: already populated in the currently committed metrics.
build_table("mono", mono_models, OUT / "baseline_mrr_small.csv")

# Expanded corpus: populated only for models whose expanded results are recorded.
build_table("expanded_indicrag_10k", expanded_models, OUT / "baseline_mrr_expanded.csv")

# Diagnostic: show exact model/setting coverage; this identifies missing results.
coverage = (
    df.groupby(["setting", "model"])
      .agg(n_languages=("key", "nunique"), languages=("key", lambda x: ",".join(sorted(set(x)))))
      .reset_index()
)
coverage.to_csv(OUT / "baseline_coverage.csv", index=False)
print("\nCoverage report saved to results/baseline_coverage.csv")
print(coverage.to_string(index=False))
