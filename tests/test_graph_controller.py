"""
Tests for GraphController - Integration tests for the controller layer.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.controllers.graph_controller import GraphController


class TestHybridChat:
    """Tests for hybrid chat functionality."""

    def test_chat_hybrid_returns_structured_response(self):
        """Test that hybrid chat returns properly structured response."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch(
                "src.controllers.graph_controller.GraphService"
            ) as mock_graph_service_class,
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            mock_graph_service = MagicMock()
            mock_graph_service_class.return_value = mock_graph_service
            mock_graph_service.query_graph_hybrid.return_value = {
                "answer": "GPT-4 is a language model.",
                "entities_found": [{"id": "GPT-4", "name": "GPT-4"}],
                "graph_context": [],
                "confidence": 0.9,
            }

            controller = GraphController()
            result = controller.chat_hybrid("What is GPT-4?")

            assert "answer" in result
            assert "entities_found" in result

    def test_chat_hybrid_handles_empty_question(self):
        """Test that empty questions are rejected."""
        with (
            patch("src.controllers.graph_controller.Neo4jRepository"),
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            controller = GraphController()
            result = controller.chat_hybrid("   ")

            assert "error" in result


class TestExploreEntity:
    """Tests for entity exploration."""

    def test_explore_existing_entity(self):
        """Test exploring an entity that exists."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch(
                "src.controllers.graph_controller.TraversalService"
            ) as mock_traversal_class,
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.vector_search.return_value = [{"id": "GPT-4", "name": "GPT-4"}]

            mock_traversal = MagicMock()
            mock_traversal_class.return_value = mock_traversal
            mock_traversal.get_entity_context.return_value = {
                "entity": {"id": "GPT-4", "name": "GPT-4"},
                "neighbors": [],
                "relationship_types": [],
            }

            controller = GraphController()
            result = controller.explore_entity("GPT-4")

            assert "entity" in result

    def test_explore_nonexistent_entity(self):
        """Test exploring an entity that doesn't exist."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.vector_search.return_value = []

            controller = GraphController()
            result = controller.explore_entity("NonExistent")

            assert result["found"] is False


class TestFindConnections:
    """Tests for connection finding."""

    def test_find_connections_between_entities(self):
        """Test finding connections between two entities."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch(
                "src.controllers.graph_controller.TraversalService"
            ) as mock_traversal_class,
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.vector_search.side_effect = [
                [{"id": "GPT-4"}],
                [{"id": "OpenAI"}],
            ]

            mock_traversal = MagicMock()
            mock_traversal_class.return_value = mock_traversal
            mock_traversal.find_connections.return_value = {
                "connected": True,
                "hops": 1,
                "description": "GPT-4 --[DEVELOPED_BY]--> OpenAI",
            }

            controller = GraphController()
            result = controller.find_connections("GPT-4", "OpenAI")

            assert result["connected"] is True


class TestModelSpecificMethods:
    """Tests for Generative AI domain-specific methods."""

    def test_get_model_info(self):
        """Test retrieving model information."""
        with (
            patch("src.controllers.graph_controller.Neo4jRepository"),
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch(
                "src.controllers.graph_controller.TraversalService"
            ) as mock_traversal_class,
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_traversal = MagicMock()
            mock_traversal_class.return_value = mock_traversal
            mock_traversal.get_model_knowledge.return_value = {
                "found": True,
                "model": {"id": "GPT-4", "name": "GPT-4"},
                "developed_by": [{"id": "OpenAI"}],
            }

            controller = GraphController()
            result = controller.get_model_info("GPT-4")

            assert result["found"] is True
            mock_traversal.get_model_knowledge.assert_called_once_with("GPT-4")

    def test_compare_models(self):
        """Test comparing two models."""
        with (
            patch("src.controllers.graph_controller.Neo4jRepository"),
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch(
                "src.controllers.graph_controller.TraversalService"
            ) as mock_traversal_class,
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_traversal = MagicMock()
            mock_traversal_class.return_value = mock_traversal
            mock_traversal.compare_models.return_value = {
                "comparable": True,
                "common_techniques": 2,
            }

            controller = GraphController()
            result = controller.compare_models("GPT-4", "Claude-3")

            assert result["comparable"] is True


class TestCypherExecution:
    """Tests for raw Cypher query execution."""

    def test_execute_cypher(self):
        """Test executing a raw Cypher query."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.execute_cypher.return_value = [{"count": 100}]

            controller = GraphController()
            result = controller.execute_cypher("MATCH (n) RETURN count(n) as count")

            assert result == [{"count": 100}]

    def test_execute_cypher_handles_errors(self):
        """Test error handling in Cypher execution."""
        with (
            patch(
                "src.controllers.graph_controller.Neo4jRepository"
            ) as mock_repo_class,
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.execute_cypher.side_effect = Exception("Syntax error")

            controller = GraphController()
            result = controller.execute_cypher("INVALID QUERY")

            assert "error" in result[0]


class TestLegacyMethods:
    """Tests for backward compatibility with legacy methods."""

    def test_legacy_chat_method(self):
        """Test the legacy chat method still works."""
        with (
            patch("src.controllers.graph_controller.Neo4jRepository"),
            patch("src.controllers.graph_controller.FileRepository"),
            patch(
                "src.controllers.graph_controller.GraphService"
            ) as mock_graph_service_class,
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            mock_graph_service = MagicMock()
            mock_graph_service_class.return_value = mock_graph_service
            mock_graph_service.query_graph.return_value = "Legacy answer"

            controller = GraphController()
            result = controller.chat("What is GPT-4?")

            assert result == "Legacy answer"

    def test_legacy_chat_rejects_empty_question(self):
        """Test legacy chat rejects empty questions."""
        with (
            patch("src.controllers.graph_controller.Neo4jRepository"),
            patch("src.controllers.graph_controller.FileRepository"),
            patch("src.controllers.graph_controller.GraphService"),
            patch("src.controllers.graph_controller.TraversalService"),
            patch("src.controllers.graph_controller.IngestService"),
        ):
            controller = GraphController()
            result = controller.chat("")

            assert "cannot be empty" in result
