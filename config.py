
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "vector_store"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval settings
TOP_K = 4
LOCAL_THRESHOLD = 0.35

# Official IIT Dharwad website pages for fallback search
SEED_URLS = [
    "https://www.iitdh.ac.in/",
    "https://www.iitdh.ac.in/academic-announcement",
    "https://www.iitdh.ac.in/academic-calendar",
    "https://www.iitdh.ac.in/academic-forms",
]

MAX_DISCOVERED_URLS = 35
