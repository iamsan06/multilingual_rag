
import pandas as pd
import pyarrow.parquet as pq
import fsspec
import re
import unicodedata
from pathlib import Path

REVISION = "72117d48b08e19a8b42ca537e483d4bb7b97d3d3"
URL = (
    "https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite/"
    f"resolve/{REVISION}/data/hi_data.parquet"
)

TARGET = 10_000
OUTPUT = Path("data/distractors/hi_distractors.parquet")
BENCHMARK = Path("data/raw/hi.parquet")


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


# Preserve all passages already in the benchmark.
benchmark = pd.read_parquet(BENCHMARK, columns=["passage"])
blocked = {
    normalize(p)
    for p in benchmark["passage"].dropna()
    if normalize(p)
}
print("Existing benchmark passages:", len(blocked))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
seen = set(blocked)
distractors = []

with fsspec.open(URL, "rb", block_size=65536) as remote:
    parquet = pq.ParquetFile(remote)
    print("Source rows:", f"{parquet.metadata.num_rows:,}")

    for i in range(parquet.metadata.num_row_groups):
        group = parquet.read_row_group(
            i, columns=["paragraph", "language"]
        ).to_pandas()

        for paragraph, language in zip(
            group["paragraph"], group["language"]
        ):
            if str(language).strip().lower() != "hi":
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

        if (i + 1) % 25 == 0:
            print(
                f"Row groups: {i + 1}/{parquet.metadata.num_row_groups} | "
                f"Unique distractors: {len(distractors)}"
            )

        if len(distractors) >= TARGET:
            break

if len(distractors) < TARGET:
    print(f"WARNING: Only collected {len(distractors)} distractors.")

pd.DataFrame({"passage": distractors}).to_parquet(
    OUTPUT, index=False
)

print("Saved:", OUTPUT)
print("Unique distractors:", len(distractors))
