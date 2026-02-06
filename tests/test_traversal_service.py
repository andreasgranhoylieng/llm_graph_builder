"""
Tests for TraversalService - Graph exploration and context extraction.
"""

from unittest.mock import MagicMock
from src.services.traversal_service import TraversalService


class TestFindConnections:
    """Tests for finding connections between entities."""

    def test_find_direct_connection(self):
        """Test finding a direct path between two entities."""
        mock_repo = MagicMock()
        mock_repo.find_path.return_value = {
            "nodes": [
                {"id": "GPT-4", "name": "GPT-4", "labels": ["AIModel"]},
                {"id": "OpenAI", "name": "OpenAI", "labels": ["AICompany"]},
            ],
            "relationships": [{"type": "DEVELOPED_BY", "properties": {}}],
        }

        service = TraversalService(mock_repo)
        result = service.find_connections("GPT-4", "OpenAI")

        assert result["connected"] is True
        assert result["hops"] == 1
        assert "DEVELOPED_BY" in result["description"]

    def test_find_no_connection(self):
        """Test when no path exists between entities."""
        mock_repo = MagicMock()
        mock_repo.find_path.return_value = None

        service = TraversalService(mock_repo)
        result = service.find_connections("EntityA", "EntityB")

        assert result["connected"] is False
        assert result["hops"] == -1
        assert "No connection found" in result["description"]

    def test_find_multi_hop_connection(self):
        """Test finding a multi-hop path."""
        mock_repo = MagicMock()
        mock_repo.find_path.return_value = {
            "nodes": [
                {"id": "GPT-4", "name": "GPT-4"},
                {"id": "OpenAI", "name": "OpenAI"},
                {"id": "Microsoft", "name": "Microsoft"},
            ],
            "relationships": [
                {"type": "DEVELOPED_BY", "properties": {}},
                {"type": "PARTNERED_WITH", "properties": {}},
            ],
        }

        service = TraversalService(mock_repo)
        result = service.find_connections("GPT-4", "Microsoft")

        assert result["connected"] is True
        assert result["hops"] == 2


class TestGetEntityContext:
    """Tests for retrieving entity context."""

    def test_get_context_for_existing_entity(self):
        """Test getting comprehensive context for an entity."""
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = {
            "id": "GPT-4",
            "name": "GPT-4",
            "description": "Large language model",
            "labels": ["AIModel"],
        }
        mock_repo.get_subgraph.return_value = {
            "nodes": [
                {"id": "GPT-4", "name": "GPT-4", "labels": ["AIModel"]},
                {"id": "OpenAI", "name": "OpenAI", "labels": ["AICompany"]},
                {"id": "doc1", "name": "Paper", "labels": ["Document"]},
            ],
            "relationships": [
                {"type": "DEVELOPED_BY", "start": "GPT-4", "end": "OpenAI"}
            ],
        }

        service = TraversalService(mock_repo)
        result = service.get_entity_context("GPT-4")

        assert result["entity"] is not None
        assert len(result["neighbors"]) == 3
        assert "DEVELOPED_BY" in result["relationship_types"]
        assert len(result["sources"]) == 1  # One Document node

    def test_get_context_for_missing_entity(self):
        """Test handling of non-existent entity."""
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = None

        service = TraversalService(mock_repo)
        result = service.get_entity_context("NonExistent")

        assert result["entity"] is None
        assert "not found" in result["context_summary"]


class TestFindSimilarEntities:
    """Tests for finding similar entities."""

    def test_find_similar_by_embedding(self):
        """Test finding similar entities using vector similarity."""
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = {
            "id": "GPT-4",
            "name": "GPT-4",
            "description": "OpenAI language model",
        }
        mock_repo.vector_search.return_value = [
            {"id": "GPT-4", "name": "GPT-4", "score": 1.0},
            {"id": "GPT-3.5", "name": "GPT-3.5", "score": 0.9},
            {"id": "Claude-3", "name": "Claude 3", "score": 0.8},
        ]

        service = TraversalService(mock_repo)
        result = service.find_similar_entities("GPT-4", top_k=2)

        # Should exclude the original entity
        assert len(result) == 2
        assert all(r["id"] != "GPT-4" for r in result)

    def test_find_similar_entity_not_found(self):
        """Test when the reference entity doesn't exist."""
        mock_repo = MagicMock()
        mock_repo.get_node_by_id.return_value = None

        service = TraversalService(mock_repo)
        result = service.find_similar_entities("NonExistent")

        assert result == []


class TestGetModelKnowledge:
    """Tests for the Generative AI domain-specific model knowledge method."""

    def test_get_comprehensive_model_info(self):
        """Test retrieving comprehensive model information."""
        mock_repo = MagicMock()
        mock_repo.vector_search.return_value = [
            {"id": "GPT-4", "name": "GPT-4", "description": "LLM", "score": 0.95}
        ]
        mock_repo.get_neighbors.return_value = [
            {
                "id": "OpenAI",
                "name": "OpenAI",
                "labels": ["AICompany"],
                "relationship": "DEVELOPED_BY",
            },
            {
                "id": "Transformer",
                "name": "Transformer",
                "labels": ["Architecture"],
                "relationship": "USES_ARCHITECTURE",
            },
            {
                "id": "RLHF",
                "name": "RLHF",
                "labels": ["Technique"],
                "relationship": "IMPLEMENTS",
            },
            {
                "id": "MMLU",
                "name": "MMLU",
                "labels": ["Benchmark"],
                "relationship": "ACHIEVES",
            },
        ]

        service = TraversalService(mock_repo)
        result = service.get_model_knowledge("GPT-4")

        assert result["found"] is True
        assert len(result["developed_by"]) == 1
        assert len(result["architecture"]) == 1
        assert len(result["techniques"]) == 1
        assert len(result["benchmarks"]) == 1

    def test_model_not_found(self):
        """Test when model is not in the knowledge graph."""
        mock_repo = MagicMock()
        mock_repo.vector_search.return_value = []

        service = TraversalService(mock_repo)
        result = service.get_model_knowledge("UnknownModel")

        assert result["found"] is False
        assert "not found" in result["message"]


class TestCompareModels:
    """Tests for model comparison functionality."""

    def test_compare_two_models(self):
        """Test comparing two AI models."""
        mock_repo = MagicMock()

        # Model A info
        mock_repo.vector_search.side_effect = [
            [{"id": "GPT-4", "name": "GPT-4", "score": 0.95}],
            [{"id": "Claude-3", "name": "Claude 3", "score": 0.95}],
        ]

        # Neighbors for each model
        mock_repo.get_neighbors.side_effect = [
            [
                {"id": "RLHF", "name": "RLHF", "labels": ["Technique"]},
                {"id": "MMLU", "name": "MMLU", "labels": ["Benchmark"]},
            ],
            [
                {"id": "RLHF", "name": "RLHF", "labels": ["Technique"]},
                {"id": "HumanEval", "name": "HumanEval", "labels": ["Benchmark"]},
            ],
        ]

        mock_repo.find_path.return_value = None  # No direct connection

        service = TraversalService(mock_repo)
        result = service.compare_models("GPT-4", "Claude-3")

        assert result["comparable"] is True
        assert result["common_techniques"] == 1  # Both use RLHF
        assert result["directly_connected"] is False

    def test_compare_with_missing_model(self):
        """Test comparison when one model doesn't exist."""
        mock_repo = MagicMock()
        mock_repo.vector_search.side_effect = [
            [{"id": "GPT-4", "name": "GPT-4", "score": 0.95}],
            [],  # Second model not found
        ]
        mock_repo.get_neighbors.return_value = []

        service = TraversalService(mock_repo)
        result = service.compare_models("GPT-4", "UnknownModel")

        assert result["comparable"] is False


class TestResearchLineage:
    """Tests for research lineage tracing."""

    def test_trace_research_lineage(self):
        """Test tracing how ideas evolved."""
        mock_repo = MagicMock()
        mock_repo.execute_cypher.return_value = [
            {
                "id": "AttentionPaper",
                "name": "Attention Is All You Need",
                "labels": ["Paper"],
                "pub_date": "2017",
            },
            {
                "id": "BERT",
                "name": "BERT",
                "labels": ["AIModel"],
                "release_date": "2018",
            },
            {"id": "GPT", "name": "GPT", "labels": ["AIModel"], "release_date": "2018"},
        ]

        service = TraversalService(mock_repo)
        result = service.get_research_lineage("Transformer")

        assert result["origin"] == "Transformer"
        assert result["count"] == 3
        assert len(result["descendants"]) == 3

    def test_lineage_handles_errors(self):
        """Test graceful handling of query errors."""
        mock_repo = MagicMock()
        mock_repo.execute_cypher.side_effect = Exception("Query failed")

        service = TraversalService(mock_repo)
        result = service.get_research_lineage("Test")

        assert result["count"] == 0
        assert "error" in result
