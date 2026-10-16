
import re
import unicodedata
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq

REVISION = "72117d48b08e19a8b42ca537e483d4bb7b97d3d3"
BASE_URL = (
    "https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite/"
    f"resolve/{REVISION}/data/"
)

LANGUAGES = [
    "as", "bn", "gu", "kn", "ml", "mr",
    "ne", "or", "pa", "ta", "te", "ur"
]
TARGET = 10_000


def normalize(value):
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", value).strip()


for lang in LANGUAGES:
    benchmark_path = Path(f"data/raw/{lang}.parquet")
    output_path = Path(f"data/distractors/{lang}_distractors.parquet")

    if not benchmark_path.exists():
        print(f"\n[{lang}] SKIPPED: benchmark file not found")
        continue

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if len(existing) == TARGET and existing["passage"].nunique() == TARGET:
            print(f"\n[{lang}] Already has 10,000 unique distractors; skipping")
            continue
        print(f"\n[{lang}] Existing output is incomplete; rebuilding")

    benchmark = pd.read_parquet(benchmark_path, columns=["passage"])
    blocked = {
        normalize(p) for p in benchmark["passage"].dropna()
        if normalize(p)
    }
    seen = set(blocked)
    distractors = []

    url = BASE_URL + f"{lang}_data.parquet"
    print(f"\n[{lang}] Starting extraction")

    try:
        with fsspec.open(url, "rb", block_size=65536) as remote:
            parquet = pq.ParquetFile(remote)
            print(f"[{lang}] Source rows: {parquet.metadata.num_rows:,}")

            for i in range(parquet.metadata.num_row_groups):
                group = parquet.read_row_group(
                    i, columns=["paragraph", "language"]
                ).to_pandas()

                for paragraph, language in zip(
                    group["paragraph"], group["language"]
                ):
                    if str(language).strip().lower() != lang:
                        continue
                    if not isinstance(paragraph, str):
                        continue

                    passage = normalize(paragraph)
                    if len(passage) < 30 or passage in seen:
                        continue

                    seen.add(passage)
                    distractors.append(passage)

                    if len(distractors) >= TARGET:
                        break

                if len(distractors) >= TARGET:
                    break

                if (i + 1) % 100 == 0:
                    print(
                        f"[{lang}] Groups: {i + 1} | "
                        f"Unique distractors: {len(distractors)}"
                    )

        if len(distractors) < TARGET:
            print(
                f"[{lang}] WARNING: only {len(distractors)} found; "
                "not saving an incomplete file"
            )
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"passage": distractors}).to_parquet(
            output_path, index=False
        )
        print(f"[{lang}] Saved {len(distractors)} unique distractors")

    except Exception as exc:
        print(f"[{lang}] ERROR: {type(exc).__name__}: {exc}")
        print(f"[{lang}] You can rerun later; existing benchmark is unchanged")

print("\nExtraction run finished.")
