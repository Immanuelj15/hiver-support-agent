"""
src/data/cleaner.py

Text cleaning and normalization utilities for social customer support messages.
Handles Twitter mentions, URL normalization, agent signature stripping, HTML entity decoding,
and contraction expansion.
"""

import re
from typing import Optional

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

def clean_tweet_text(text: Optional[str], is_brand_reply: bool = False) -> str:
    """
    Clean raw tweet text:
    1. Strip Twitter @handles (e.g. @AmazonHelp, @115820)
    2. Expand common conversational contractions
    3. Normalize shortened URLs to standard token [link]
    4. Decode HTML entities (&amp;, &lt;, &gt;)
    5. Strip human agent sign-offs (^MD, ^SJ, -John, ET) from brand replies
    6. Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    # Strip Twitter handles
    cleaned = re.sub(r"@\w+", "", cleaned)

    # Expand contractions
    for pattern, replacement in CONTRACTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Strip agent sign-off boilerplate if it is a brand reply
    if is_brand_reply:
        cleaned = re.sub(r"\^[A-Za-z]{1,4}\s*$", "", cleaned)
        cleaned = re.sub(r"\b[A-Z]{2}\s*$", "", cleaned)
        cleaned = re.sub(r"-[A-Za-z]+\s*$", "", cleaned)

    # Normalize URLs
    cleaned = re.sub(r"https?://(?:t\.co|\S+)/\S*", "[link]", cleaned)

    # Decode HTML entities
    cleaned = (
        cleaned.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
