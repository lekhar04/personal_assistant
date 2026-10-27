"""Central configuration. Everything can be overridden with environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "chatbot.db"
CHROMA_DIR = DATA_DIR / "chroma"

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Provider: "gemini" or "anthropic". If unset, it is picked from whichever API key is present.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or (
    "gemini" if GEMINI_API_KEY and not ANTHROPIC_API_KEY else "anthropic"
)
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-flash-lite-latest"}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "claude-sonnet-5-5")

def llm_configured() -> bool:
    key = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else ANTHROPIC_API_KEY
    return bool(key)


def missing_key_message() -> str:
    name = "GEMINI_API_KEY" if LLM_PROVIDER == "gemini" else "ANTHROPIC_API_KEY"
    return f"{name} is not set. Add it to your .env file and restart the app."
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "1500"))
RESPONSE_TEMPERATURE = float(os.getenv("RESPONSE_TEMPERATURE", "0.4"))

# --- Short-term memory (conversation history) ---
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

# --- Long-term memory (cosine distance: 0 = identical, 2 = opposite) ---
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "4"))
MEMORY_MAX_DISTANCE = float(os.getenv("MEMORY_MAX_DISTANCE", "0.70"))
MEMORY_DEDUP_DISTANCE = float(os.getenv("MEMORY_DEDUP_DISTANCE", "0.15"))

# --- Document RAG ---
DOC_TOP_K = int(os.getenv("DOC_TOP_K", "5"))
DOC_MAX_DISTANCE = float(os.getenv("DOC_MAX_DISTANCE", "0.75"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

DEFAULT_CHAT_TITLE = "New chat"
