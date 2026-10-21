# Tokenizer fertility experiment (Phase 1)
# Run from repository root after copying/placing expanded Parquet files under data/processed/<lang>/.
# Produces per-language fertility metrics and correlation against EXPANDED-corpus MRR only.

from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from common import LANGS, DATA_PROC

OUT = Path("tokenizer/out")
OUT.mkdir(parents=True, exist_ok=True)

E5_NAME = "intfloat/multilingual-e5-base"
BGE_NAME = "BAAI/bge-m3"
MAX_PASSAGES = 5000

def nfc(text):
    return unicodedata.normalize("NFC", " ".join(str(text).split()))

def fertility(tokenizer, texts):
    texts = [nfc(t) for t in texts[:MAX_PASSAGES]]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    n_tokens = sum(len(ids) for ids in encoded)
    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    if n_words == 0 or n_tokens == 0:
        raise ValueError("No words/tokens found in sample.")
    return n_tokens / n_words, n_chars / n_tokens

e5 = AutoTokenizer.from_pretrained(E5_NAME)
bge = AutoTokenizer.from_pretrained(BGE_NAME)
print("Tokenizer vocab sizes:", len(e5), len(bge))

rows = []
for lang in LANGS:
    corpus_path = DATA_PROC / lang / "corpus_expanded.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing {corpus_path}. Copy/download the expanded corpus first; "
            "do not run on the small corpus by accident."
        )
    corpus = pd.read_parquet(corpus_path)
    if "text" not in corpus.columns:
        raise ValueError(f"{corpus_path} needs a 'text' column; found {list(corpus.columns)}")
    passages = corpus["text"].astype(str).tolist()
    e5_tpw, e5_cpt = fertility(e5, passages)
    bge_tpw, bge_cpt = fertility(bge, passages)
    rows.append({
        "lang": lang,
        "n_passages_sampled": min(len(passages), MAX_PASSAGES),
        "e5_tokens_per_word": e5_tpw,
        "e5_chars_per_token": e5_cpt,
        "bge_tokens_per_word": bge_tpw,
        "bge_chars_per_token": bge_cpt,
    })
    print(lang, f"E5={e5_tpw:.3f} tok/word", f"BGE={bge_tpw:.3f} tok/word")

fert = pd.DataFrame(rows).sort_values("lang")
fert.to_csv(OUT / "fertility.csv", index=False, float_format="%.6f")
print(f"\nSaved {OUT / 'fertility.csv'}")

# Use only expanded-corpus metric rows. The retrieval score order has to be query/corpus aligned.
metrics_path = Path("results/metrics.csv")
metrics = pd.read_csv(metrics_path)
metrics = metrics.loc[metrics["setting"].eq("expanded_indicrag_10k")].copy()
model_candidates = {
    "bm25": "bm25_expanded",
    "bge_m3": "bge-m3-expanded",
    "e5_small": "mE5-small-expanded",
    "e5_base": "mE5-base-expanded",
    "e5_large": "mE5-large-expanded",
}
metrics["lang"] = metrics["key"]
wide = metrics.pivot(index="lang", columns="model", values="MRR")
wide = wide.reindex(LANGS)

merged = fert.set_index("lang").join(
    wide.rename(columns={v: k for k, v in model_candidates.items()}),
    how="left"
)
merged.to_csv(OUT / "fertility_with_mrr.csv", float_format="%.6f")

# Correlations are calculated only when a model has at least 5 languages recorded.
corr_rows = []
for model in model_candidates:
    if model not in merged.columns:
        continue
    valid = merged[["e5_tokens_per_word", model]].dropna()
    if len(valid) < 5:
        print(f"Skipping {model}: only {len(valid)} expanded-language result(s)")
        continue
    rho, p = spearmanr(valid["e5_tokens_per_word"], valid[model])
    corr_rows.append({"model": model, "n_languages": len(valid), "spearman_rho": rho, "p_value": p})

corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "fertility_correlations.csv", index=False, float_format="%.6f")
print("\nCorrelation results:")
print(corr.to_string(index=False) if not corr.empty else "No model has enough matching expanded results yet.")

# Plot BGE-M3 if its expanded MRR is present for multiple languages; otherwise plot available baseline.
plot_model = next((m for m in ["bge_m3", "e5_base", "bm25"] if m in merged.columns and merged[m].notna().sum() >= 2), None)
if plot_model:
    valid = merged[["e5_tokens_per_word", plot_model]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["e5_tokens_per_word"], valid[plot_model])
    for lang, row in valid.iterrows():
        ax.annotate(lang, (row["e5_tokens_per_word"], row[plot_model]))
    ax.set_xlabel("mE5-base tokens per whitespace word")
    ax.set_ylabel(f"Expanded-corpus MRR ({plot_model})")
    ax.set_title("Tokenizer fertility vs retrieval performance")
    fig.tight_layout()
    fig.savefig(OUT / "fertility_vs_mrr.png", dpi=200)
    plt.close(fig)
    print(f"Saved {OUT / 'fertility_vs_mrr.png'}")
else:
    print("Plot skipped: not enough expanded-corpus MRR values to correlate.")
