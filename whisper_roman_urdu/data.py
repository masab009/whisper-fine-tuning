import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio


def load_roman_urdu_dataset(
    tsv_path: str,
    audio_base_dir: str,
    test_size: float = 0.02,
    seed: int = 42,
) -> DatasetDict:
    df = pd.read_csv(tsv_path, sep="\t")

    # required columns
    if "path" not in df.columns or "roman_sentence" not in df.columns:
        raise ValueError("TSV must contain columns: path, roman_sentence")

    # Clean text
    df = df[["path", "roman_sentence"]].dropna()
    df["roman_sentence"] = df["roman_sentence"].astype(str).str.strip()
    df = df[df["roman_sentence"] != ""]

    # FIX: mp3 → wav (matches your current dataset naming)
    df["path"] = df["path"].astype(str).str.replace(".mp3", ".wav", regex=False)

    # Build absolute paths
    df["path"] = df["path"].apply(lambda p: os.path.join(audio_base_dir, p))

    # Filter missing audio
    exists_mask = df["path"].apply(os.path.isfile)
    missing = int((~exists_mask).sum())
    if missing > 0:
        print(f"⚠️ Dropping {missing} samples with missing audio files")

    df = df[exists_mask].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("❌ No audio files found after path normalization")

    # Build HF dataset
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.rename_columns({"path": "audio", "roman_sentence": "text"})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Train/validation split
    if len(ds) < 100:
        print("⚠️ Small dataset — skipping validation split")
        return DatasetDict(train=ds, validation=ds.select([]))

    splits = ds.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(train=splits["train"], validation=splits["test"])


def add_labels_column(batch, tokenizer, max_length: int):
    batch["labels"] = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    ).input_ids
    return batch
