"""
CTI Knowledge Graph Extraction & Benchmarking Framework
Central Configuration Module

All project-wide constants, paths, model configurations,
and runtime settings are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_DIR = PROJECT_ROOT / "CTI_Report_Dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
PROMPT_DIR = PROJECT_ROOT / "prompts"
CACHE_DIR = PROJECT_ROOT / "cache"

# ─── Ensure Directories Exist ────────────────────────────────────────────────
for _dir in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Runtime Mode ────────────────────────────────────────────────────────────
DEV_MODE = True  # Set to False for production runs
MAX_EVENTS_DEV = 5  # Max events to process in dev mode

# ─── Checkpoint Configuration ────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N events

# ─── Rate Limiting ───────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds (exponential backoff base)
REQUEST_TIMEOUT = 300  # seconds (generous for local Ollama models)

# ─── Model Generation Settings (centralized) ────────────────────────────────
TEMPERATURE = 0.1  # Low temperature for structured extraction output
TEMPERATURE_RAW = 0.3  # Slightly higher for free-form generation
MAX_OUTPUT_TOKENS = 4096  # Max tokens for JSON output generation

# ─── API Keys (from .env) ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "") or os.getenv("GORQ_API_KEY", "")

# ─── Neo4j Configuration (from .env) ─────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ─── Ollama Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = "gemma_e2b:latest"

# ─── Model Registry ──────────────────────────────────────────────────────────
# Maps user-friendly names to implementation details
MODEL_REGISTRY = {
    "gemini": {
        "class": "GeminiLLM",
        "module": "models.gemini_model",
        "api_key_env": "GEMINI_API_KEY",
    },
    "mistral": {
        "class": "MistralLLM",
        "module": "models.mistral_model",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "llama_groq": {
        "class": "GroqLLM",
        "module": "models.groq_model",
        "api_key_env": "GROQ_API_KEY",
        "model_id": "llama-3.1-8b-instant",
    },
    "gpt_oss_groq": {
        "class": "GroqLLM",
        "module": "models.groq_model",
        "api_key_env": "GROQ_API_KEY",
        "model_id": "gpt-oss-20b",
    },
    "ollama_gemma": {
        "class": "OllamaLLM",
        "module": "models.ollama_model",
        "model_id": "gemma_e2b:latest",
    },
}

# ─── Retriever Registry ──────────────────────────────────────────────────────
RETRIEVER_REGISTRY = {
    "llm_only": {
        "class": "LLMOnlyRetriever",
        "module": "retrievers.llm_only",
    },
    "vanilla_rag": {
        "class": "VanillaRAGRetriever",
        "module": "retrievers.vanilla_rag",
    },
    "graph_rag": {
        "class": "GraphRAGRetriever",
        "module": "retrievers.graph_rag",
    },
}

# ─── Benchmark Matrix ────────────────────────────────────────────────────────
BENCHMARK_MODELS = ["gemini", "mistral", "llama_groq", "gpt_oss_groq"]
BENCHMARK_RETRIEVERS = ["llm_only", "vanilla_rag", "graph_rag"]

# ─── Prompt Files ─────────────────────────────────────────────────────────────
EXTRACTION_PROMPT_FILE = PROMPT_DIR / "extraction_prompt.txt"
EVALUATION_PROMPT_FILE = PROMPT_DIR / "evaluation_prompt.txt"

# ─── Logging Configuration ───────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─── Git Hash (for reproducibility tracking) ─────────────────────────────────
try:
    import subprocess
    GIT_HASH = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(PROJECT_ROOT),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    GIT_HASH = "unknown"
