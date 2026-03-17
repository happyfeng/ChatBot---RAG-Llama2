# -*- coding: utf-8 -*-
"""Configuration for PDF + Llama2 chatbot (RAG & fine-tuning)."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / os.getenv("PDF_DIR", "pdfs")
VECTOR_STORE_DIR = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "vector_store")
CHROMA_PERSIST_DIR = str(VECTOR_STORE_DIR / "chroma")

# Model names (Hugging Face)
LLAMA2_MODEL = os.getenv("LLAMA2_MODEL", "meta-llama/Llama-2-7b-chat-hf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Offline: set to 1 when network unavailable; ensure embedding model is local (cache or EMBEDDING_MODEL points to local dir)
_def = os.getenv("HF_HUB_OFFLINE", "").strip().lower()
_local_embed = EMBEDDING_MODEL and (Path(EMBEDDING_MODEL).is_dir() or (PROJECT_ROOT / EMBEDDING_MODEL).is_dir())
if _def in ("1", "true", "yes") or _local_embed:
    os.environ["HF_HUB_OFFLINE"] = "1"
if _local_embed and not Path(EMBEDDING_MODEL).is_absolute():
    EMBEDDING_MODEL = str(PROJECT_ROOT / EMBEDDING_MODEL)

# Fine-tuning
FINETUNE_OUTPUT_DIR = PROJECT_ROOT / os.getenv("FINETUNE_OUTPUT_DIR", "finetuned_llama")
FINETUNE_DATA_DIR = PROJECT_ROOT / os.getenv("FINETUNE_DATA_DIR", "finetune_data")

# Hugging Face token (required for gated Llama2; set in .env)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Ollama (local inference; alternative to Hugging Face)
USE_OLLAMA = os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def ensure_dirs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
