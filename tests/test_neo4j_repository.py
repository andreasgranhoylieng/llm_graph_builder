"""
Tests for Neo4jRepository - Vector search, graph traversal, and data operations.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.repositories.neo4j_repository import Neo4jRepository


class TestNeo4jRepositoryConnection:
    """Tests for database connection handling."""

    def test_connection_error_handling(self, mock_neo4j_graph):
        """Test that connection errors are properly wrapped."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as mock:
            mock.side_effect = Exception("Unable to retrieve routing information")

            repo = Neo4jRepository()

            with pytest.raises(ConnectionError) as exc_info:
                repo._get_graph()

            assert "Could not connect to Neo4j" in str(exc_info.value)

    def test_lazy_connection_initialization(self):
        """Test that connection is not created until needed."""
        repo = Neo4jRepository()
        assert repo._graph is None


class TestVectorSearch:
    """Tests for vector search functionality."""

    def test_vector_search_returns_results(
        self, mock_neo4j_graph, mock_embeddings, sample_entities
    ):
        """Test that vector search returns properly formatted results."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
        ):
            # Setup mocks
            graph_instance = MagicMock()
            neo4j_mock.return_value = graph_instance
            graph_instance.query.return_value = sample_entities

            emb_instance = MagicMock()
            emb_mock.return_value = emb_instance
            emb_instance.embed_query.return_value = [0.1] * 3072

            repo = Neo4jRepository()
            results = repo.vector_search("GPT-4 language model", top_k=3)

            # Verify
            assert len(results) == 3
            assert results[0]["id"] == "GPT-4"
            emb_instance.embed_query.assert_called_once()

    def test_vector_search_with_label_filter(self, mock_neo4j_graph, mock_embeddings):
        """Test vector search with node label filtering."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
        ):
            neo4j_mock.return_value = mock_neo4j_graph
            emb_mock.return_value = mock_embeddings
            mock_neo4j_graph.query.return_value = []

            repo = Neo4jRepository()
            repo.vector_search("test", node_labels=["AIModel", "AICompany"])

            # Check that the query included label filter
            call_args = mock_neo4j_graph.query.call_args
            assert call_args is not None

    def test_vector_search_handles_empty_results(
        self, mock_neo4j_graph, mock_embeddings
    ):
        """Test vector search returns empty list when no matches."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
        ):
            neo4j_mock.return_value = mock_neo4j_graph
            emb_mock.return_value = mock_embeddings
            mock_neo4j_graph.query.return_value = []

            repo = Neo4jRepository()
            results = repo.vector_search("nonexistent entity")

            assert results == []


class TestGraphTraversal:
    """Tests for graph traversal functionality."""

    def test_get_node_by_id_found(self, mock_neo4j_graph):
        """Test retrieving a node by ID."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = [
                {
                    "n": {"id": "GPT-4", "name": "GPT-4", "description": "LLM"},
                    "labels": ["AIModel"],
                }
            ]

            repo = Neo4jRepository()
            result = repo.get_node_by_id("GPT-4")

            assert result is not None
            assert result["id"] == "GPT-4"
            assert result["labels"] == ["AIModel"]

    def test_get_node_by_id_not_found(self, mock_neo4j_graph):
        """Test retrieving a non-existent node."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = []

            repo = Neo4jRepository()
            result = repo.get_node_by_id("NonExistent")

            assert result is None

    def test_get_neighbors(self, mock_neo4j_graph, sample_neighbors):
        """Test getting neighboring nodes."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = sample_neighbors

            repo = Neo4jRepository()
            results = repo.get_neighbors("GPT-4", depth=1)

            assert len(results) == 2
            assert results[0]["relationship"] == "DEVELOPED_BY"

    def test_get_neighbors_with_direction(self, mock_neo4j_graph):
        """Test getting outgoing neighbors only."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = []

            repo = Neo4jRepository()
            repo.get_neighbors("GPT-4", depth=1, direction="out")

            # Verify query was called
            mock_neo4j_graph.query.assert_called()

    def test_find_path(self, mock_neo4j_graph):
        """Test finding path between two nodes."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = [
                {
                    "nodes": [
                        {"id": "GPT-4", "name": "GPT-4", "labels": ["AIModel"]},
                        {"id": "OpenAI", "name": "OpenAI", "labels": ["AICompany"]},
                    ],
                    "relationships": [{"type": "DEVELOPED_BY", "properties": {}}],
                }
            ]

            repo = Neo4jRepository()
            result = repo.find_path("GPT-4", "OpenAI")

            assert result is not None
            assert len(result["nodes"]) == 2
            assert result["relationships"][0]["type"] == "DEVELOPED_BY"


class TestHybridQuery:
    """Tests for hybrid query functionality."""

    def test_hybrid_query_with_matches(
        self, mock_neo4j_graph, mock_embeddings, sample_entities, sample_neighbors
    ):
        """Test hybrid query returns structured response."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
            patch("src.repositories.neo4j_repository.ChatOpenAI") as llm_mock,
        ):
            neo4j_mock.return_value = mock_neo4j_graph
            emb_mock.return_value = mock_embeddings

            # First call returns entities (vector search)
            # Second call returns neighbors
            # vector search returns entities, doc search returns [],
            # then parallel BFS seed expansions and reranking queries follow
            mock_neo4j_graph.query.side_effect = [
                sample_entities,  # vector search
                [],  # document search
                Exception("APOC"),  # seed 1 APOC attempt
                [
                    {
                        "nodes": [
                            {
                                "id": "GPT-4",
                                "name": "GPT-4",
                                "labels": ["AIModel"],
                                "description": "",
                            }
                        ]
                    }
                ],  # seed 1 fallback
                Exception("APOC"),  # seed 2 APOC attempt
                [
                    {
                        "nodes": [
                            {
                                "id": "GPT-3.5",
                                "name": "GPT-3.5",
                                "labels": ["AIModel"],
                                "description": "",
                            }
                        ]
                    }
                ],  # seed 2 fallback
                Exception("APOC"),  # seed 3 APOC attempt
                [
                    {
                        "nodes": [
                            {
                                "id": "Claude-3",
                                "name": "Claude-3",
                                "labels": ["AIModel"],
                                "description": "",
                            }
                        ]
                    }
                ],  # seed 3 fallback
                [
                    {"eid": "GPT-4", "min_distance": 0},
                    {"eid": "GPT-3.5", "min_distance": 1},
                    {"eid": "Claude-3", "min_distance": 2},
                ],  # reranking
            ]

            mock_embeddings.embed_query.return_value = [0.1] * 3072

            llm_instance = MagicMock()
            llm_mock.return_value = llm_instance
            mock_response = MagicMock()
            mock_response.content = "GPT-4 is a language model by OpenAI."
            llm_instance.invoke.return_value = mock_response

            repo = Neo4jRepository()
            result = repo.query_hybrid("What is GPT-4?")

            assert "answer" in result
            assert "entities_found" in result
            assert result["method"] == "hybrid_sota"

    def test_hybrid_query_falls_back_to_cypher(self, mock_neo4j_graph, mock_embeddings):
        """Test hybrid query falls back to Cypher when no vector matches."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
            patch("src.repositories.neo4j_repository.GraphCypherQAChain") as chain_mock,
        ):
            neo4j_mock.return_value = mock_neo4j_graph
            emb_mock.return_value = mock_embeddings

            # No vector search results
            mock_neo4j_graph.query.return_value = []
            mock_embeddings.embed_query.return_value = [0.1] * 3072

            chain_instance = MagicMock()
            chain_mock.from_llm.return_value = chain_instance
            chain_instance.invoke.return_value = {"result": "Cypher answer"}

            repo = Neo4jRepository()
            result = repo.query_hybrid("Unknown topic")

            assert result["method"] == "cypher_fallback"


class TestEmbeddingGeneration:
    """Tests for embedding generation during ingestion."""

    def test_embeddings_use_name_and_description(self, sample_graph_document):
        """Test that embeddings are generated from name + description."""
        with (
            patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock,
            patch("src.repositories.neo4j_repository.OpenAIEmbeddings") as emb_mock,
        ):
            mock_graph = MagicMock()
            neo4j_mock.return_value = mock_graph

            mock_embeddings = MagicMock()
            emb_mock.return_value = mock_embeddings
            mock_embeddings.embed_documents.return_value = [[0.1] * 3072, [0.1] * 3072]

            repo = Neo4jRepository()
            repo.add_graph_documents_batch([sample_graph_document])

            # Verify embed_documents was called
            assert mock_embeddings.embed_documents.called

            # Get the texts that were embedded
            call_args = mock_embeddings.embed_documents.call_args
            texts = call_args[0][0]

            # Should have embedded at least one text
            assert len(texts) > 0

            # The embedding should include the node name
            assert any("GPT-4" in text for text in texts)


class TestDatabaseManagement:
    """Tests for database management operations."""

    def test_create_indexes(self, mock_neo4j_graph):
        """Test that index creation queries are executed."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph

            repo = Neo4jRepository()
            repo.create_indexes()

            # Should have called query multiple times for different indexes
            assert mock_neo4j_graph.query.call_count > 5

    def test_get_statistics(self, mock_neo4j_graph):
        """Test retrieving database statistics."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.side_effect = [
                [{"count": 100}],  # nodes
                [{"count": 50}],  # relationships
                [{"labels": ["AIModel"], "count": 30}],  # label counts
                [{"count": 80}],  # embedded count
            ]

            repo = Neo4jRepository()
            stats = repo.get_statistics()

            assert stats["total_nodes"] == 100
            assert stats["total_relationships"] == 50

    def test_clear_database_requires_confirmation(self, mock_neo4j_graph):
        """Test that clear_database requires explicit confirmation."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph

            repo = Neo4jRepository()

            with pytest.raises(ValueError) as exc_info:
                repo.clear_database(confirm=False)

            assert "confirm=True" in str(exc_info.value)


class TestBidirectionalBFS:
    """Tests for the SOTA bidirectional BFS implementation."""

    def test_bfs_finds_path_via_apoc(self, mock_neo4j_graph):
        """Test that APOC-based BFS returns a path when available."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.return_value = [
                {
                    "nodes": [
                        {"id": "A", "name": "A", "labels": ["Entity"]},
                        {"id": "B", "name": "B", "labels": ["Entity"]},
                    ],
                    "relationships": [{"type": "RELATED_TO", "properties": {}}],
                }
            ]

            repo = Neo4jRepository()
            result = repo.bfs_search("A", "B", max_depth=5)

            assert result is not None
            assert len(result["nodes"]) == 2
            assert result["nodes"][0]["id"] == "A"

    def test_bfs_fallback_to_simple_path(self, mock_neo4j_graph):
        """Test that BFS falls back to simple path when APOC and bidirectional fail."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            # First call (APOC) fails, second (bidirectional) fails, third (simple) succeeds
            mock_neo4j_graph.query.side_effect = [
                Exception("APOC not available"),
                Exception("Bidirectional failed"),
                [
                    {
                        "nodes": [
                            {"id": "X", "name": "X", "labels": ["Entity"]},
                            {"id": "Y", "name": "Y", "labels": ["Entity"]},
                        ],
                        "relationships": [{"type": "LINKS_TO", "properties": {}}],
                    }
                ],
            ]

            repo = Neo4jRepository()
            result = repo.bfs_search("X", "Y")

            assert result is not None
            assert len(result["nodes"]) == 2

    def test_bfs_returns_none_when_no_path(self, mock_neo4j_graph):
        """Test that BFS returns None when no path exists."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.side_effect = [
                Exception("APOC"),
                Exception("Bidir"),
                [],
            ]

            repo = Neo4jRepository()
            result = repo.bfs_search("isolated_a", "isolated_b")

            assert result is None


class TestParallelBFS:
    """Tests for multi-source parallel BFS."""

    def test_parallel_bfs_merges_subgraphs(self, mock_neo4j_graph):
        """Test that parallel BFS merges subgraphs from multiple seeds."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            # Each seed expansion returns different nodes
            # APOC fails, fallback returns nodes
            mock_neo4j_graph.query.side_effect = [
                Exception("APOC"),  # seed 1 APOC attempt
                [
                    {
                        "nodes": [
                            {
                                "id": "A",
                                "name": "A",
                                "labels": ["E"],
                                "description": "",
                            },
                            {
                                "id": "C",
                                "name": "C",
                                "labels": ["E"],
                                "description": "bridge",
                            },
                        ]
                    }
                ],
                Exception("APOC"),  # seed 2 APOC attempt
                [
                    {
                        "nodes": [
                            {
                                "id": "B",
                                "name": "B",
                                "labels": ["E"],
                                "description": "",
                            },
                            {
                                "id": "C",
                                "name": "C",
                                "labels": ["E"],
                                "description": "bridge",
                            },
                        ]
                    }
                ],
            ]

            repo = Neo4jRepository()
            result = repo.parallel_bfs_from_seeds(["A", "B"], max_depth=2)

            # Should have 3 unique nodes (A, B, C)
            node_ids = {n["id"] for n in result["nodes"]}
            assert "A" in node_ids
            assert "B" in node_ids
            assert "C" in node_ids
            assert len(node_ids) == 3

    def test_parallel_bfs_identifies_bridge_nodes(self, mock_neo4j_graph):
        """Test that bridge nodes (found from 2+ seeds) are correctly identified."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.side_effect = [
                Exception("APOC"),
                [
                    {
                        "nodes": [
                            {
                                "id": "seed1",
                                "name": "S1",
                                "labels": ["E"],
                                "description": "",
                            },
                            {
                                "id": "bridge",
                                "name": "Bridge",
                                "labels": ["E"],
                                "description": "shared",
                            },
                        ]
                    }
                ],
                Exception("APOC"),
                [
                    {
                        "nodes": [
                            {
                                "id": "seed2",
                                "name": "S2",
                                "labels": ["E"],
                                "description": "",
                            },
                            {
                                "id": "bridge",
                                "name": "Bridge",
                                "labels": ["E"],
                                "description": "shared",
                            },
                        ]
                    }
                ],
            ]

            repo = Neo4jRepository()
            result = repo.parallel_bfs_from_seeds(["seed1", "seed2"], max_depth=2)

            bridge_ids = {b["id"] for b in result["bridge_nodes"]}
            assert "bridge" in bridge_ids
            # Seeds themselves should NOT be in bridge_nodes
            assert "seed1" not in bridge_ids
            assert "seed2" not in bridge_ids

    def test_parallel_bfs_empty_seeds(self, mock_neo4j_graph):
        """Test that empty seed list returns empty result."""
        repo = Neo4jRepository()
        result = repo.parallel_bfs_from_seeds([])

        assert result["nodes"] == []
        assert result["relationships"] == []
        assert result["bridge_nodes"] == []

    def test_parallel_bfs_deduplicates_seeds(self, mock_neo4j_graph):
        """Test that duplicate seed IDs are deduplicated."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.side_effect = [
                Exception("APOC"),
                [
                    {
                        "nodes": [
                            {"id": "A", "name": "A", "labels": ["E"], "description": ""}
                        ]
                    }
                ],
            ]

            repo = Neo4jRepository()
            result = repo.parallel_bfs_from_seeds(["A", "A", "A"], max_depth=2)

            # Should only expand once (deduplicated)
            assert len(result["nodes"]) == 1


class TestGraphRerank:
    """Tests for graph-aware reranking."""

    def test_reranking_boosts_connected_entities(self, mock_neo4j_graph):
        """Test that entities closer in the graph get higher blended scores."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            # Return proximity distances
            mock_neo4j_graph.query.return_value = [
                {"eid": "close_entity", "min_distance": 1},
                {"eid": "far_entity", "min_distance": 5},
            ]

            entities = [
                {"id": "far_entity", "score": 0.9, "name": "Far"},
                {"id": "close_entity", "score": 0.8, "name": "Close"},
            ]

            repo = Neo4jRepository()
            result = repo.graph_rerank(
                entities,
                query_entity_ids=["query_node"],
                vector_weight=0.6,
                graph_weight=0.4,
            )

            # Close entity (distance=1, proximity=0.833) should rank higher
            # despite lower vector score
            # close: 0.6*0.8 + 0.4*0.833 = 0.48 + 0.333 = 0.813
            # far:   0.6*0.9 + 0.4*0.167 = 0.54 + 0.067 = 0.607
            assert result[0]["id"] == "close_entity"
            assert result[0]["graph_proximity"] > result[1]["graph_proximity"]

    def test_reranking_preserves_entities_on_failure(self, mock_neo4j_graph):
        """Test that reranking returns original entities if query fails."""
        with patch("src.repositories.neo4j_repository.Neo4jGraph") as neo4j_mock:
            neo4j_mock.return_value = mock_neo4j_graph
            mock_neo4j_graph.query.side_effect = Exception("Query failed")

            entities = [
                {"id": "A", "score": 0.9, "name": "A"},
                {"id": "B", "score": 0.8, "name": "B"},
            ]

            repo = Neo4jRepository()
            result = repo.graph_rerank(entities, query_entity_ids=["Q"])

            # Should return original entities unchanged
            assert len(result) == 2
            assert result[0]["id"] == "A"

    def test_reranking_empty_input(self, mock_neo4j_graph):
        """Test that empty inputs are handled gracefully."""
        repo = Neo4jRepository()

        assert repo.graph_rerank([], ["Q"]) == []
        assert repo.graph_rerank([{"id": "A", "score": 0.9}], []) == [
            {"id": "A", "score": 0.9}
        ]
