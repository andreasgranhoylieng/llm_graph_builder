import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and DB Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# LLM Configuration
LLM_MODEL = "gpt-4o-mini-2024-07-18"  # or gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini
LLM_TEMPERATURE = 0.0

# --- SCHEMA DEFINITION ---
# This restricts the LLM to specific types, preventing graph chaos.
ALLOWED_NODES = [
    "Person",
    "Organization",
    "Technology",
    "Event",
    "Product",
    "Concept",
    "Location",
]

ALLOWED_RELATIONSHIPS = [
    "WORKS_AT",
    "DEVELOPED",
    "RELEASED",
    "ACQUIRED",
    "PART_OF",
    "LOCATED_AT",
    "RELATED_TO",
]

# =============================================================================
# LARGE-SCALE PROCESSING CONFIGURATION
# =============================================================================

# Batch Processing Configuration
BATCH_SIZE_FILES = 10  # Files per batch
BATCH_SIZE_CHUNKS = 50  # Chunks per LLM batch
MAX_CONCURRENT_LLM_CALLS = 5  # Parallel LLM requests

# Rate Limiting (OpenAI tier-dependent - adjust based on your tier)
RATE_LIMIT_RPM = 500  # Requests per minute
RATE_LIMIT_TPM = 150000  # Tokens per minute

# Checkpointing
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 10  # Save state every N files

# Memory Management
MAX_MEMORY_MB = 4096  # Memory limit before forcing batch flush

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier

# Logging
LOG_DIR = "./logs"
ENABLE_DETAILED_LOGGING = True
