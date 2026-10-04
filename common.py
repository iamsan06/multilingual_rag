import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

LANGS = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr",
    "ne", "or", "pa", "ta", "te", "ur"
]

HF_DATASET = "ai4bharat/IndicMSMARCO"

SEED = 42

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
EMB_DIR = ROOT / "embeddings"
RUN_DIR = ROOT / "runs"
RES_DIR = ROOT / "results"

for _d in (
    DATA_RAW,
    DATA_PROC,
    EMB_DIR,
    RUN_DIR,
    RES_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Paper Table 2 reference values
# ----------------------------------------------------------------------------

# Order:
# e5-small, e5-base, e5-large, LLM2Vec, BGE-M3

PAPER_TABLE2 = {
    "as": (0.30, 0.40, 0.45, 0.42, 0.46),
    "bn": (0.39, 0.46, 0.48, 0.44, 0.49),
    "gu": (0.34, 0.43, 0.48, 0.42, 0.48),
    "hi": (0.44, 0.49, 0.52, 0.49, 0.52),
    "kn": (0.38, 0.44, 0.47, 0.40, 0.47),
    "ml": (0.38, 0.45, 0.49, 0.43, 0.49),
    "mr": (0.36, 0.45, 0.49, 0.45, 0.49),
    "ne": (0.39, 0.45, 0.49, 0.45, 0.49),
    "or": (0.31, 0.39, 0.45, 0.34, 0.45),
    "pa": (0.32, 0.42, 0.48, 0.42, 0.48),
    "ta": (0.38, 0.45, 0.49, 0.40, 0.49),
    "te": (0.39, 0.45, 0.50, 0.42, 0.50),
    "ur": (0.35, 0.45, 0.49, 0.44, 0.48),
}

PAPER_COLS = [
    "mE5-small",
    "mE5-base",
    "mE5-large",
    "LLM2Vec",
    "BGE-M3",
]


# ----------------------------------------------------------------------------
# Text / IDs
# ----------------------------------------------------------------------------

def norm_text(s) -> str:
    """
    NFC-normalise and collapse whitespace.

    Used for:
    - stable document IDs
    - deduplication
    - text passed to the encoder
    """

    s = unicodedata.normalize(
        "NFC",
        str(s),
    )

    return re.sub(
        r"\s+",
        " ",
        s,
    ).strip()


def text_id(s) -> str:
    """
    Content-based passage ID.

    This is NOT a row index.
    """

    return (
        "p_"
        + hashlib.sha1(
            norm_text(s).encode("utf-8")
        ).hexdigest()[:16]
    )


# ----------------------------------------------------------------------------
# Data loading / benchmark construction
# ----------------------------------------------------------------------------

def load_raw(lang: str) -> pd.DataFrame:
    """
    Load one language config of IndicMSMARCO.

    The first call downloads from Hugging Face and caches
    the language as Parquet under data/raw/.
    """

    if lang not in LANGS:
        raise ValueError(
            f"Unknown language '{lang}'. "
            f"Expected one of {LANGS}"
        )

    p = DATA_RAW / f"{lang}.parquet"

    if p.exists():
        return pd.read_parquet(p)

    from datasets import load_dataset

    ds = load_dataset(
        HF_DATASET,
        lang,
        split="train",
    )

    df = ds.to_pandas()

    df.to_parquet(
        p,
        index=False,
    )

    return df


def build_benchmark(
    df: pd.DataFrame,
    seed: int = SEED,
):
    """
    Build the benchmark from one language.

    corpus:
        unique passages
        doc_id = content hash
        order shuffled

    queries:
        query_id
        query
        gold_doc_id

    IMPORTANT:
    Only `query` and `passage` are used.

    `text`, `answer`, `title`, `url`, and `meta`
    are deliberately excluded to prevent leakage.
    """

    d = df.copy()

    d["query_id"] = d["query_id"].astype(str)

    d["query_n"] = d["query"].map(norm_text)

    d["passage_n"] = d["passage"].map(norm_text)

    # Remove empty queries/passages
    d = d[
        (d["query_n"] != "")
        &
        (d["passage_n"] != "")
    ]

    # One query per query_id
    d = d.drop_duplicates(
        subset=["query_id"],
        keep="first",
    )

    # Content-based document ID
    d["doc_id"] = d["passage_n"].map(text_id)

    # Build corpus
    corpus = (
        d.drop_duplicates("doc_id")[
            ["doc_id", "passage_n"]
        ]
        .rename(
            columns={
                "passage_n": "text"
            }
        )
        .sample(
            frac=1.0,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    # Build query table
    queries = (
        d[
            [
                "query_id",
                "query_n",
                "doc_id",
            ]
        ]
        .rename(
            columns={
                "query_n": "query",
                "doc_id": "gold_doc_id",
            }
        )
        .reset_index(drop=True)
    )

    return queries, corpus


def save_benchmark(
    lang,
    queries,
    corpus,
):
    """
    Save processed benchmark files.
    """

    out = DATA_PROC / lang

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    queries.to_parquet(
        out / "queries.parquet",
        index=False,
    )

    corpus.to_parquet(
        out / "corpus.parquet",
        index=False,
    )


def load_benchmark(lang, seed=SEED):
    """
    Load a processed benchmark.

    If the processed benchmark does not exist yet,
    automatically build it from IndicMSMARCO and save it.

    Returns
    -------
    queries : pd.DataFrame
    corpus : pd.DataFrame
    gold : np.ndarray

        gold[i] is the corpus row index containing
        the relevant document for query i.
    """

    if lang not in LANGS:
        raise ValueError(
            f"Unknown language '{lang}'. "
            f"Expected one of {LANGS}"
        )

    out = DATA_PROC / lang

    query_path = out / "queries.parquet"
    corpus_path = out / "corpus.parquet"

    # --------------------------------------------------------
    # Build benchmark automatically if it doesn't exist
    # --------------------------------------------------------

    if not query_path.exists() or not corpus_path.exists():

        print(
            f"[{lang}] Processed benchmark not found."
        )

        print(
            f"[{lang}] Loading from Hugging Face..."
        )

        df = load_raw(lang)

        queries, corpus = build_benchmark(
            df,
            seed=seed,
        )

        save_benchmark(
            lang,
            queries,
            corpus,
        )

        print(
            f"[{lang}] Benchmark created:"
        )

        print(
            f"  Queries: {len(queries)}"
        )

        print(
            f"  Corpus:  {len(corpus)}"
        )

    # --------------------------------------------------------
    # Load processed benchmark
    # --------------------------------------------------------

    queries = pd.read_parquet(
        query_path
    )

    corpus = pd.read_parquet(
        corpus_path
    )

    # --------------------------------------------------------
    # Map gold document IDs to corpus row indices
    # --------------------------------------------------------

    id2idx = {
        doc_id: idx
        for idx, doc_id
        in enumerate(corpus["doc_id"])
    }

    gold = queries["gold_doc_id"].map(
        id2idx
    )

    # --------------------------------------------------------
    # Critical invariant
    # --------------------------------------------------------

    assert gold.notna().all(), (
        f"[{lang}] Some gold document IDs "
        "are missing from the corpus."
    )

    assert queries["query_id"].is_unique

    assert corpus["doc_id"].is_unique

    return (
        queries,
        corpus,
        gold.to_numpy(dtype=np.int64),
    )

def gold_ranks(S, gold):

    S = np.asarray(
        S,
        dtype=np.float32,
    )

    g = S[
        np.arange(len(gold)),
        gold,
    ][:, None]

    greater = (
        S > g
    ).sum(1)

    ties_other = (
        S == g
    ).sum(1) - 1

    return (
        1
        + greater
        + ties_other
    ).astype(np.int64)


def metrics_from_ranks(
    r,
    ks=(1, 5, 10, 20, 100),
):
    
    r = np.asarray(r)

    out = {
        "MRR": float(
            np.mean(1.0 / r)
        ),

        "MRR@10": float(
            np.mean(
                np.where(
                    r <= 10,
                    1.0 / r,
                    0.0,
                )
            )
        ),
    }

    for k in ks:
        out[f"R@{k}"] = float(
            np.mean(r <= k)
        )

    out["nDCG@10"] = float(
        np.mean(
            np.where(
                r <= 10,
                1.0 / np.log2(r + 1),
                0.0,
            )
        )
    )

    out["n_queries"] = int(
        len(r)
    )

    return out

def bootstrap_ci(
    values,
    n_boot=2000,
    seed=0,
):
    
    rng = np.random.default_rng(
        seed
    )

    v = np.asarray(
        values,
        dtype=np.float64,
    )

    idx = rng.integers(
        0,
        len(v),
        size=(
            n_boot,
            len(v),
        ),
    )

    means = v[idx].mean(axis=1)

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def paired_bootstrap_diff(
    a,
    b,
    n_boot=2000,
    seed=0,
):
    
    d = (
        np.asarray(
            a,
            dtype=np.float64,
        )
        -
        np.asarray(
            b,
            dtype=np.float64,
        )
    )

    lo, hi = bootstrap_ci(
        d,
        n_boot,
        seed,
    )

    return (
        float(d.mean()),
        lo,
        hi,
    )

def record(
    model,
    key,
    S,
    gold,
    setting="mono",
):
    
    S = np.asarray(
        S,
        dtype=np.float32,
    )

    r = gold_ranks(
        S,
        gold,
    )

    d = RUN_DIR / model

    d.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        d / f"{key}_scores.npy",
        S,
    )

    np.save(
        d / f"{key}_ranks.npy",
        r,
    )

    m = metrics_from_ranks(r)

    m.update(
        model=model,
        key=key,
        setting=setting,
        n_docs=int(
            S.shape[1]
        ),
    )

    path = RES_DIR / "metrics.csv"

    row = pd.DataFrame([m])

    if path.exists():

        old = pd.read_csv(path)

        keep = ~(
            (old.model == model)
            &
            (old.key == key)
            &
            (old.setting == setting)
        )

        row = pd.concat(
            [
                old[keep],
                row,
            ],
            ignore_index=True,
        )

    row.to_csv(
        path,
        index=False,
    )

    return m


def load_scores(
    model,
    key,
):
    """
    Load a previously saved score matrix.
    """

    return np.load(
        RUN_DIR
        / model
        / f"{key}_scores.npy"
    )

def pool_sensitivity(
    S,
    gold,
    sizes=(100, 250, 500, 1000),
    reps=5,
    seed=0,
):
    
    rng = np.random.default_rng(
        seed
    )

    nq, nd = S.shape

    rows = []

    for n in sizes:

        n = min(
            n,
            nd,
        )

        mrrs = []

        for _ in range(reps):

            rr = np.empty(
                nq
            )

            for q in range(nq):

                others = np.delete(
                    np.arange(nd),
                    gold[q],
                )

                pick = rng.choice(
                    others,
                    size=n - 1,
                    replace=False,
                )

                rr[q] = (
                    1
                    + (
                        S[
                            q,
                            pick,
                        ]
                        >= S[
                            q,
                            gold[q],
                        ]
                    ).sum()
                )

            mrrs.append(
                np.mean(1.0 / rr)
            )

        rows.append(
            (
                n,
                float(
                    np.mean(mrrs)
                ),
                float(
                    np.std(mrrs)
                ),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "pool_size",
            "MRR",
            "std_over_reps",
        ],
    )


# ----------------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------------

def tokenize(text) -> list:
    """
    Script-safe tokenizer.

    DO NOT use:
        re.findall(r'\\w+', text)

    because Indic combining vowel signs / matras
    can be split incorrectly.

    Processing:
    - NFC
    - lowercase
    - remove ZWJ/ZWNJ
    - punctuation/symbol/separator/control -> spaces
    """

    t = unicodedata.normalize(
        "NFC",
        str(text),
    ).lower()

    t = (
        t
        .replace("\u200c", "")
        .replace("\u200d", "")
    )

    t = "".join(
        " "
        if unicodedata.category(c)[0]
        in ("P", "S", "Z", "C")
        else c
        for c in t
    )

    return t.split()


class BM25:

    def __init__(
        self,
        k1=1.5,
        b=0.75,
    ):
        self.k1 = k1
        self.b = b

    def fit(
        self,
        docs_tokens,
    ):
        """
        Fit BM25 on tokenized documents.
        """

        vocab = {}
        rows = []
        cols = []
        vals = []

        for i, toks in enumerate(
            docs_tokens
        ):

            cnt = {}

            for t in toks:
                cnt[t] = (
                    cnt.get(t, 0)
                    + 1
                )

            for t, c in cnt.items():

                j = vocab.setdefault(
                    t,
                    len(vocab),
                )

                rows.append(i)
                cols.append(j)
                vals.append(c)

        N = len(
            docs_tokens
        )

        tf = sparse.csr_matrix(
            (
                vals,
                (rows, cols),
            ),
            shape=(
                N,
                len(vocab),
            ),
            dtype=np.float64,
        )

        dl = np.array(
            [
                len(t)
                for t in docs_tokens
            ],
            dtype=np.float64,
        )

        avgdl = dl.mean()

        df = np.bincount(
            tf.indices,
            minlength=len(vocab),
        )

        idf = np.log(
            1.0
            + (
                N
                - df
                + 0.5
            )
            / (
                df
                + 0.5
            )
        )

        coo = tf.tocoo()

        denom = (
            coo.data
            + self.k1
            * (
                1
                - self.b
                + self.b
                * dl[coo.row]
                / avgdl
            )
        )

        w = (
            coo.data
            * (self.k1 + 1)
            / denom
            * idf[coo.col]
        )

        self.W = sparse.csr_matrix(
            (
                w,
                (
                    coo.row,
                    coo.col,
                ),
            ),
            shape=tf.shape,
        )

        self.vocab = vocab

        return self

    def score(
        self,
        queries_tokens,
    ):
        """
        Score every query against every document.

        Returns:
            (n_queries, n_docs)
        """

        rows = []
        cols = []
        vals = []

        for i, toks in enumerate(
            queries_tokens
        ):

            for t in toks:

                j = self.vocab.get(t)

                if j is not None:

                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0)

        Q = sparse.csr_matrix(
            (
                vals,
                (rows, cols),
            ),
            shape=(
                len(queries_tokens),
                len(self.vocab),
            ),
            dtype=np.float64,
        )

        return (
            Q @ self.W.T
        ).toarray()


# ----------------------------------------------------------------------------
# Dense encoders
# ----------------------------------------------------------------------------

def encode_cached(
    model,
    texts,
    cache_path,
    prefix="",
    batch_size=16,
    normalize=True,
):
    """
    Sentence-Transformers encoding with
    on-disk caching.
    """

    cache_path = Path(
        cache_path
    )

    if cache_path.exists():
        return np.load(
            cache_path
        )

    emb = model.encode(
        [
            prefix + t
            for t in texts
        ],
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb = emb.astype(
        np.float32
    )

    np.save(
        cache_path,
        emb,
    )

    return emb


def encode_hf_meanpool(
    tokenizer,
    model,
    texts,
    cache_path,
    batch_size=16,
    max_length=256,
    prefix="",
):
    
    import torch

    cache_path = Path(
        cache_path
    )

    if cache_path.exists():
        return np.load(
            cache_path
        )

    texts = [
        prefix + t
        for t in texts
    ]

    # Long texts first -> less padding waste
    order = np.argsort(
        [
            -len(t)
            for t in texts
        ]
    )

    out = np.zeros(
        (
            len(texts),
            model.config.hidden_size,
        ),
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():

        for s in range(
            0,
            len(texts),
            batch_size,
        ):

            ids = order[
                s:s + batch_size
            ]

            enc = tokenizer(
                [
                    texts[i]
                    for i in ids
                ],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            h = model(
                **enc
            ).last_hidden_state

            m = (
                enc["attention_mask"]
                .unsqueeze(-1)
                .to(h.dtype)
            )

            e = (
                (h * m).sum(1)
                /
                m.sum(1).clamp(
                    min=1
                )
            )

            e = (
                torch.nn.functional.normalize(
                    e,
                    dim=-1,
                )
            )

            out[ids] = (
                e.cpu().numpy()
            )

    np.save(
        cache_path,
        out,
    )

    return out


# ----------------------------------------------------------------------------
# Fusion / reranking helpers
# ----------------------------------------------------------------------------

def rrf_fuse(
    score_mats,
    k=60,
    depth=None,
):
    
    nq, nd = score_mats[0].shape

    depth = (
        nd
        if depth is None
        else min(depth, nd)
    )

    fused = np.zeros(
        (
            nq,
            nd,
        ),
        dtype=np.float64,
    )

    contrib = (
        1.0
        /
        (
            k
            + np.arange(
                1,
                depth + 1,
            )
        )
    )

    for S in score_mats:

        order = np.argsort(
            -S,
            axis=1,
            kind="stable",
        )[:, :depth]

        for q in range(nq):

            fused[
                q,
                order[q],
            ] += contrib

    return fused


def topk_candidates(
    S,
    k=100,
):
    
    return np.argsort(
        -S,
        axis=1,
        kind="stable",
    )[:, :k]


def merge_rerank(
    S_base,
    cand,
    ce_scores,
):
    
    nq, nd = S_base.shape

    base_rank = (
        np.argsort(
            np.argsort(
                -S_base,
                axis=1,
                kind="stable",
            ),
            axis=1,
        )
        + 1
    )

    out = (
        -base_rank.astype(
            np.float64
        )
    )

    for q in range(nq):

        out[
            q,
            cand[q],
        ] = (
            1e6
            + ce_scores[q]
        )

    return out