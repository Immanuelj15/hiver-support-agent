"""
src/ingest.py

Loads Kaggle Customer Support on Twitter CSV (twcs.csv), filters for a specific
brand (default: AmazonHelp), reconstructs conversation threads, cleans text, and
outputs data/processed/threads.jsonl.
"""

import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Common contraction expansion dictionary
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bn't\b": " not",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\byou're\b": "you are",
    r"\byou've\b": "you have",
    r"\byou'll\b": "you will",
    r"\byou'd\b": "you would",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bwhat's\b": "what is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bwhere's\b": "where is",
    r"\bwho's\b": "who is",
    r"\bhow's\b": "how is",
    r"\blet's\b": "let us",
}

def clean_text(text: str, is_brand_reply: bool = False) -> str:
    """Clean Twitter text: handles, contractions, urls, and agent signatures."""
    if not isinstance(text, str):
        return ""
    
    cleaned = text
    
    # 1. Strip leading and embedded Twitter handles (e.g. @AmazonHelp, @115820)
    cleaned = re.sub(r"@\w+", "", cleaned)
    
    # 2. Expand contractions (case-insensitive)
    for pattern, replacement in CONTRACTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    # 3. Clean agent sign-off boilerplate if brand reply (e.g., ^TN, ^SJ, ^MD, ET, -John)
    if is_brand_reply:
        cleaned = re.sub(r"\^[A-Za-z]{1,4}\s*$", "", cleaned)
        cleaned = re.sub(r"\b[A-Z]{2}\s*$", "", cleaned)
        cleaned = re.sub(r"-[A-Za-z]+\s*$", "", cleaned)
    
    # 4. Normalize URLs (shortened t.co links)
    cleaned = re.sub(r"https?://t\.co/\S+", "[link]", cleaned)
    
    # 5. Clean extra whitespace, quotes, html entities
    cleaned = cleaned.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned

def is_predominantly_english(text: str) -> bool:
    """Filter out non-English messages using ASCII & Latin printable ratio."""
    if not text:
        return False
    # Check ratio of ascii printable characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / max(len(text), 1)) > 0.85

def reconstruct_threads(
    csv_path: str,
    brand: str = "AmazonHelp",
    sample_size: int = None,
    output_path: str = "data/processed/threads.jsonl"
):
    print(f"Loading data from {csv_path} (sample_size={sample_size})...")
    
    usecols = [
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id"
    ]
    
    df = pd.read_csv(csv_path, nrows=sample_size, usecols=usecols, low_memory=False)
    print(f"Loaded {len(df):,} total tweets.")
    
    # Map tweet_id to row for fast parent/follow-up lookups
    print("Building tweet index...")
    # Convert tweet_id and in_response_to_tweet_id to numeric
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df = df.dropna(subset=["tweet_id"])
    df["tweet_id"] = df["tweet_id"].astype(np.int64)
    
    # Filter brand replies
    brand_df = df[(df["author_id"] == brand) & (~df["inbound"])]
    print(f"Found {len(brand_df):,} replies authored by brand '{brand}'.")
    
    # Fast dictionary lookup for customer tweets
    tweet_dict = {}
    for row in df.itertuples(index=False):
        tweet_dict[row.tweet_id] = row
        
    threads = []
    print("Reconstructing conversation threads...")
    
    for brand_row in brand_df.itertuples(index=False):
        parent_id = brand_row.in_response_to_tweet_id
        if pd.isna(parent_id):
            continue
        try:
            parent_id = int(float(parent_id))
        except (ValueError, TypeError):
            continue
            
        if parent_id not in tweet_dict:
            continue
            
        cust_row = tweet_dict[parent_id]
        
        # Only consider inbound customer messages
        if not cust_row.inbound:
            continue
            
        cust_raw = cust_row.text
        brand_raw = brand_row.text
        
        # Filter for predominantly English
        if not (is_predominantly_english(cust_raw) and is_predominantly_english(brand_raw)):
            continue
            
        cust_cleaned = clean_text(cust_raw, is_brand_reply=False)
        brand_cleaned = clean_text(brand_raw, is_brand_reply=True)
        
        # Filter out trivially short messages (e.g. < 4 words)
        if len(cust_cleaned.split()) < 4 or len(brand_cleaned.split()) < 4:
            continue
            
        # Check for optional customer follow-up
        followup_cleaned = None
        followup_id = brand_row.response_tweet_id
        if pd.notna(followup_id):
            # Sometimes multiple response IDs are comma separated
            first_resp_id = str(followup_id).split(",")[0].strip()
            try:
                first_resp_id = int(float(first_resp_id))
                if first_resp_id in tweet_dict:
                    f_row = tweet_dict[first_resp_id]
                    if f_row.inbound:
                        followup_cleaned = clean_text(f_row.text, is_brand_reply=False)
            except (ValueError, TypeError):
                pass
                
        threads.append({
            "thread_id": f"{cust_row.tweet_id}_{brand_row.tweet_id}",
            "customer_msg": cust_cleaned,
            "brand_reply": brand_cleaned,
            "customer_followup": followup_cleaned,
            "timestamp": str(brand_row.created_at)
        })
        
    print(f"Successfully reconstructed {len(threads):,} clean conversation threads.")
    
    # Save to JSONL
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")
            
    print(f"Saved to {output_path}")
    
    # Print basic statistics
    if threads:
        cust_lens = [len(t["customer_msg"].split()) for t in threads]
        timestamps = [t["timestamp"] for t in threads if t["timestamp"]]
        print("\n--- Ingestion Statistics ---")
        print(f"Total reconstructed threads: {len(threads):,}")
        print(f"Median customer_msg length (words): {int(np.median(cust_lens))}")
        print(f"Mean customer_msg length (words): {np.mean(cust_lens):.1f}")
        print(f"With customer follow-up: {sum(1 for t in threads if t['customer_followup']):,} ({sum(1 for t in threads if t['customer_followup'])/len(threads)*100:.1f}%)")
        if timestamps:
            print(f"Date range: {min(timestamps)} to {max(timestamps)}")
        print("----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and reconstruct brand support threads.")
    parser.add_argument("--csv_path", type=str, default="data/raw/twcs.csv", help="Path to twcs.csv")
    parser.add_argument("--brand", type=str, default="AmazonHelp", help="Target brand author_id")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of rows to sample from CSV")
    parser.add_argument("--output_path", type=str, default="data/processed/threads.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    
    reconstruct_threads(
        csv_path=args.csv_path,
        brand=args.brand,
        sample_size=args.sample_size,
        output_path=args.output_path
    )
