import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean-like environment values."""
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}

# =============================================================================
# API KEYS AND DATABASE CONFIGURATION
# =============================================================================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv(
    "AZURE_OPENAI_END_POINT"
)
AZURE_OPENAI_API_VERSION = (
    os.getenv("AZURE_OPENAI_API_VERSION")
    or os.getenv("OPEN_API_VERSION_35")
    or "2024-10-21"
)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
LLM_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
    or os.getenv("DEPLOYMENT_NAME")
    or LLM_MODEL
)
LLM_TEMPERATURE = 0.0

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL")
    or os.getenv("EMBEDDING_MODEL_NAME")
    or "text-embedding-3-large"
)
EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    os.getenv("EMBEDDING_DEPLOYMENT_NAME", EMBEDDING_MODEL),
)
_embedding_dimension_default = 3072
if EMBEDDING_MODEL in {"text-embedding-3-small", "text-embedding-ada-002"}:
    _embedding_dimension_default = 1536
EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", str(_embedding_dimension_default))
)

# =============================================================================
# GENERATIVE AI KNOWLEDGE GRAPH SCHEMA
# =============================================================================
# This schema is optimized for Generative AI domain knowledge extraction.
# It captures AI models, companies, research, datasets, and their relationships.

ALLOWED_NODES: List[str] = [
    # Core AI Entities
    "AIModel",  # GPT-4, Claude, Llama, Gemini, etc.
    "AICompany",  # OpenAI, Anthropic, Google DeepMind, Meta AI, etc.
    "Researcher",  # Key researchers and scientists
    "Paper",  # Research papers and publications
    "Dataset",  # Training datasets (ImageNet, Common Crawl, etc.)
    # Technical Components
    "Architecture",  # Transformer, Diffusion, GAN, etc.
    "Technique",  # RLHF, RAG, LoRA, Quantization, etc.
    "Benchmark",  # MMLU, HumanEval, GLUE, etc.
    "Framework",  # PyTorch, TensorFlow, JAX, LangChain, etc.
    # Applications & Products
    "Application",  # ChatGPT, Copilot, Midjourney, etc.
    "UseCase",  # Code generation, Image synthesis, etc.
    "Feature",  # Context window, Function calling, Vision, etc.
    # Domain & Context
    "Concept",  # Abstract concepts (attention, emergence, hallucination)
    "Organization",  # Non-AI companies, universities, labs
    "Event",  # Conferences, releases, announcements
    "License",  # MIT, Apache 2.0, Proprietary, etc.
]

ALLOWED_RELATIONSHIPS: List[str] = [
    # Creation & Development
    "DEVELOPED_BY",  # Model -> Company/Researcher
    "CREATED",  # Company/Researcher -> Model/Paper
    "AUTHORED",  # Researcher -> Paper
    "TRAINED_ON",  # Model -> Dataset
    "FINE_TUNED_FROM",  # Model -> Model (base model relationship)
    # Technical Architecture
    "USES_ARCHITECTURE",  # Model -> Architecture
    "IMPLEMENTS",  # Model/Framework -> Technique
    "BASED_ON",  # Architecture/Model -> Paper/Concept
    "EXTENDS",  # Technique -> Technique
    "REQUIRES",  # Model -> Framework/Technique
    # Performance & Evaluation
    "EVALUATED_ON",  # Model -> Benchmark
    "OUTPERFORMS",  # Model -> Model (with score property)
    "ACHIEVES",  # Model -> Benchmark (with score property)
    # Products & Applications
    "POWERS",  # Model -> Application
    "HAS_FEATURE",  # Model/Application -> Feature
    "SUPPORTS",  # Framework -> Technique/Feature
    "ENABLES",  # Technique -> UseCase
    # Organizational
    "AFFILIATED_WITH",  # Researcher -> Organization/Company
    "ACQUIRED",  # Company -> Company
    "PARTNERED_WITH",  # Company -> Company
    "FUNDED_BY",  # Company -> Organization
    "RELEASED_AT",  # Model/Paper -> Event
    # Licensing & Access
    "LICENSED_UNDER",  # Model/Dataset -> License
    # Semantic Relationships
    "RELATED_TO",  # General semantic connection
    "SIMILAR_TO",  # Similar models/techniques
    "CONTRASTS_WITH",  # Opposing approaches
    "PART_OF",  # Component relationships
    "INSTANCE_OF",  # Type hierarchies
    "SUCCEEDED_BY",  # Temporal succession
]

# =============================================================================
# NODE PROPERTY SCHEMA
# =============================================================================
# Defines required and optional properties for each node type.
# This ensures consistent data structure for embeddings and queries.

NODE_PROPERTY_SCHEMA: Dict[str, Dict[str, Any]] = {
    "__Entity__": {
        # Base properties for ALL entities
        "required": ["id", "name"],
        "optional": [
            "description",
            "aliases",
            "source_document",
            "source_chunk",
            "confidence",
            "created_at",
        ],
        "embedding_fields": [
            "name",
            "description",
        ],  # Fields used for vector embeddings
    },
    "AIModel": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "release_date",
            "parameters",
            "context_length",
            "training_tokens",
            "modality",
            "open_source",
            "version",
            "aliases",
            "source_document",
            "source_chunk",
            "confidence",
        ],
        "embedding_fields": ["name", "description"],
    },
    "AICompany": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "founded_date",
            "headquarters",
            "valuation",
            "employee_count",
            "website",
            "aliases",
            "source_document",
        ],
        "embedding_fields": ["name", "description"],
    },
    "Researcher": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "affiliation",
            "h_index",
            "notable_work",
            "aliases",
            "source_document",
        ],
        "embedding_fields": ["name", "description", "notable_work"],
    },
    "Paper": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "abstract",
            "publication_date",
            "venue",
            "citation_count",
            "arxiv_id",
            "doi",
            "aliases",
            "source_document",
        ],
        "embedding_fields": ["name", "description", "abstract"],
    },
    "Architecture": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "introduced_in",
            "key_innovation",
            "aliases",
            "source_document",
        ],
        "embedding_fields": ["name", "description", "key_innovation"],
    },
    "Technique": {
        "required": ["id", "name"],
        "optional": [
            "description",
            "category",
            "introduced_in",
            "aliases",
            "source_document",
        ],
        "embedding_fields": ["name", "description"],
    },
}

# =============================================================================
# RELATIONSHIP PROPERTY SCHEMA
# =============================================================================
RELATIONSHIP_PROPERTIES: Dict[str, List[str]] = {
    "OUTPERFORMS": ["benchmark", "score_difference", "metric"],
    "ACHIEVES": ["score", "metric", "date"],
    "TRAINED_ON": ["data_percentage", "tokens_used"],
    "FINE_TUNED_FROM": ["method", "dataset_used"],
    # All relationships can have these
    "__default__": ["source_chunk_text", "source_file", "confidence", "extracted_at"],
}

# =============================================================================
# LARGE-SCALE PROCESSING CONFIGURATION
# =============================================================================
BATCH_SIZE_FILES = 10
BATCH_SIZE_CHUNKS = 50
MAX_CONCURRENT_LLM_CALLS = 5
GRAPH_BATCHES_PER_WORKER = int(os.getenv("GRAPH_BATCHES_PER_WORKER", "2"))
GRAPH_MIN_BATCHES_PER_FILE = int(os.getenv("GRAPH_MIN_BATCHES_PER_FILE", "8"))
GRAPH_LOG_RATE_LIMIT_WAIT = _env_bool("GRAPH_LOG_RATE_LIMIT_WAIT", True)

# Ingestion quality/performance tuning
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "2000"))
INGEST_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))
INGEST_MIN_CHUNK_CHARS = int(os.getenv("INGEST_MIN_CHUNK_CHARS", "1"))
INGEST_NORMALIZE_WHITESPACE = _env_bool("INGEST_NORMALIZE_WHITESPACE", True)
INGEST_DEDUP_WITHIN_FILE = _env_bool("INGEST_DEDUP_WITHIN_FILE", True)
INGEST_DEDUP_ACROSS_JOB = _env_bool("INGEST_DEDUP_ACROSS_JOB", True)

# Database write/embedding throughput tuning
NEO4J_INGEST_BATCH_SIZE = int(os.getenv("NEO4J_INGEST_BATCH_SIZE", "100"))
GRAPH_WRITE_FLUSH_SIZE = int(os.getenv("GRAPH_WRITE_FLUSH_SIZE", "300"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "96"))
SKIP_EXISTING_EMBEDDINGS = _env_bool("SKIP_EXISTING_EMBEDDINGS", True)

# Graph extraction detail/latency tradeoff
GRAPH_EXTRACT_NODE_PROPERTIES = _env_bool("GRAPH_EXTRACT_NODE_PROPERTIES", True)
GRAPH_EXTRACT_RELATIONSHIP_PROPERTIES = _env_bool(
    "GRAPH_EXTRACT_RELATIONSHIP_PROPERTIES", True
)

# Rate Limiting (OpenAI tier-dependent)
RATE_LIMIT_RPM = 500
RATE_LIMIT_TPM = 150000

# Checkpointing
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 10

# Memory Management
MAX_MEMORY_MB = 4096

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
RETRY_BACKOFF_MULTIPLIER = 2

# Logging
LOG_DIR = "./logs"
ENABLE_DETAILED_LOGGING = True

# =============================================================================
# VECTOR SEARCH CONFIGURATION
# =============================================================================
VECTOR_SEARCH_TOP_K = 30
VECTOR_SEARCH_SCORE_THRESHOLD = 0.7
HYBRID_SEARCH_DEPTH = 10  # Hops for graph expansion after vector search

# =============================================================================
# SOTA GRAPH RAG CONFIGURATION
# =============================================================================
MAX_PARALLEL_BFS_WORKERS = 3  # Concurrent BFS threads for multi-source search
BFS_MAX_DEPTH = 4  # Max hops for BFS traversals
RERANK_VECTOR_WEIGHT = 0.6  # Weight for vector similarity in reranking
RERANK_GRAPH_WEIGHT = 0.4  # Weight for graph proximity in reranking
MULTI_HOP_CONTEXT_DEPTH = 3  # Depth for multi-hop context extraction
