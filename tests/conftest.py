"""
Pytest configuration and shared fixtures for the LLM Graph Builder test suite.
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_neo4j_graph():
    """Mock Neo4jGraph for testing without database connection."""
    with patch("src.repositories.neo4j_repository.Neo4jGraph") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Set up default return values
        mock_instance.query.return_value = []
        mock_instance.add_graph_documents.return_value = None
        mock_instance.refresh_schema.return_value = None

        yield mock_instance


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings for testing without API calls."""
    with patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Return fake embeddings (3072 dimensions for text-embedding-3-large)
        fake_embedding = [0.1] * 3072
        mock_instance.embed_query.return_value = fake_embedding
        mock_instance.embed_documents.return_value = [fake_embedding]

        yield mock_instance


@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI for testing without API calls."""
    with patch("src.services.graph_service.ChatOpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "This is a test answer."
        mock_instance.invoke.return_value = mock_response

        yield mock_instance


@pytest.fixture
def sample_graph_document():
    """Create a sample graph document for testing."""
    from langchain_core.documents import Document

    class MockNode:
        def __init__(self, id: str, type: str, properties: dict = None):
            self.id = id
            self.type = type
            self.properties = properties or {}

    class MockRelationship:
        def __init__(self, source: MockNode, target: MockNode, type: str):
            self.source = source
            self.target = target
            self.type = type
            self.properties = {}

    class MockGraphDocument:
        def __init__(self):
            self.nodes = [
                MockNode("GPT-4", "AIModel", {"description": "Large language model"}),
                MockNode("OpenAI", "AICompany", {"description": "AI research company"}),
            ]
            self.relationships = [
                MockRelationship(self.nodes[0], self.nodes[1], "DEVELOPED_BY")
            ]
            self.source = Document(
                page_content="GPT-4 is a large language model developed by OpenAI.",
                metadata={"source": "test.pdf", "id": "test-doc-1"},
            )

    return MockGraphDocument()


@pytest.fixture
def sample_documents():
    """Create sample documents for testing ingestion."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="GPT-4 is OpenAI's most advanced model, featuring improved reasoning.",
            metadata={"source": "test1.pdf", "page": 1},
        ),
        Document(
            page_content="Claude is Anthropic's flagship AI assistant, trained with RLHF.",
            metadata={"source": "test2.pdf", "page": 1},
        ),
    ]


# =============================================================================
# TEST DATA
# =============================================================================


@pytest.fixture
def sample_entities() -> List[Dict[str, Any]]:
    """Sample entity data for vector search testing."""
    return [
        {
            "id": "GPT-4",
            "name": "GPT-4",
            "description": "OpenAI's large language model with multimodal capabilities",
            "labels": ["AIModel", "__Entity__"],
            "score": 0.95,
        },
        {
            "id": "GPT-3.5",
            "name": "GPT-3.5",
            "description": "Earlier version of GPT",
            "labels": ["AIModel", "__Entity__"],
            "score": 0.85,
        },
        {
            "id": "Claude-3",
            "name": "Claude 3",
            "description": "Anthropic's constitutional AI model",
            "labels": ["AIModel", "__Entity__"],
            "score": 0.75,
        },
    ]


@pytest.fixture
def sample_neighbors() -> List[Dict[str, Any]]:
    """Sample neighbor data for traversal testing."""
    return [
        {
            "id": "OpenAI",
            "name": "OpenAI",
            "description": "AI research company",
            "labels": ["AICompany", "__Entity__"],
            "relationship": "DEVELOPED_BY",
        },
        {
            "id": "Transformer",
            "name": "Transformer",
            "description": "Attention-based architecture",
            "labels": ["Architecture", "__Entity__"],
            "relationship": "USES_ARCHITECTURE",
        },
    ]


# =============================================================================
# INTEGRATION TEST MARKERS
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real services"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set environment variables for testing."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "test-password")
