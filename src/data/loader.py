"""
src/data/loader.py

Memory-safe chunked loader for Kaggle Customer Support on Twitter (twcs.csv)
and structured JSONL thread loader.
"""

import os
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import pandas as pd

from src.data.cleaner import clean_tweet_text

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    records = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON on line {line_num} in {file_path}: {e}")
    return records

def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """Save a list of dictionaries as JSONL."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def stream_twcs_chunks(
    csv_path: str,
    chunksize: int = 100_000,
    target_author: Optional[str] = None
) -> Iterator[pd.DataFrame]:
    """
    Stream chunks from twcs.csv to avoid out-of-memory errors on large datasets.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV not found at {csv_path}")

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        if target_author:
            # Filter chunk for either messages by target author or referencing target author
            author_mask = chunk["author_id"].astype(str) == str(target_author)
            text_mask = chunk["text"].astype(str).str.contains(f"@{target_author}", case=False, na=False)
            yield chunk[author_mask | text_mask]
        else:
            yield chunk
