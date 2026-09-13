import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "twcs.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
THREADS_PATH = PROCESSED_DATA_DIR / "threads.jsonl"
INTENTS_LABELED_PATH = PROCESSED_DATA_DIR / "intents_labeled.jsonl"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "embeddings_cache.npz"
GOLDEN_SET_PATH = BASE_DIR / "golden_set" / "golden_150.jsonl"
EVAL_RESULTS_PATH = BASE_DIR / "eval" / "results.json"
EVAL_REPORT_PATH = BASE_DIR / "eval" / "report.md"

# Brand settings
TARGET_BRAND = "AmazonHelp"

# Model settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.55
TOP_K_RETRIEVAL = 3

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else ("ollama" if os.getenv("USE_OLLAMA") else "gemini"))
if LLM_PROVIDER.lower() == "ollama" or os.getenv("USE_OLLAMA", "").lower() in ("true", "1", "yes"):
    LLM_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "mistral:latest"))
else:
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")

# Intent taxonomy (derived from historical data)
INTENTS = [
    "order_delivery_delay",       # late delivery, tracking not updating, carrier issues
    "damaged_or_wrong_item",      # broken goods, missing parts, received incorrect product
    "return_and_refund",          # returns processing, refund confirmation/timelines
    "account_security_and_login", # password reset, 2FA/OTP issues, hacked account
    "subscription_and_billing",   # Prime membership renewal, unexpected credit card charges
    "product_technical_issue",    # Kindle/Fire TV/Echo setup, app crashes, digital item
    "feedback_or_complaint",      # venting about customer service, packaging, delivery agent
    "other"                       # out-of-scope, ambiguous, legal threats, policy edge cases
]

# Sensitive intents requiring immediate escalation
SENSITIVE_INTENTS = {
    "account_security_and_login",
    "other"
}

def get_llm_client(provider: str = None):
    """Returns an OpenAI-compatible client configured for Ollama, Gemini, or OpenAI."""
    from openai import OpenAI
    
    active_provider = (provider or os.getenv("LLM_PROVIDER", "")).lower()
    use_ollama = os.getenv("USE_OLLAMA", "").lower() in ("true", "1", "yes")
    
    if active_provider == "ollama" or use_ollama:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(
            base_url=ollama_url,
            api_key="ollama"
        )
        
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key and active_provider != "openai":
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_key
        )
    elif openai_key:
        return OpenAI(api_key=openai_key)
    else:
        # Fallback to local Ollama if no cloud keys
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
