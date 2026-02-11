"""
Tests for GraphService - Graph extraction and document processing.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.services.graph_service import GraphService


class TestGraphExtraction:
    """Tests for graph extraction from documents."""

    def test_extract_entity_description(self):
        """Test extracting description from source text."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        source_text = "GPT-4 is a large language model developed by OpenAI. It features improved reasoning capabilities."

        description = service._extract_entity_description("GPT-4", source_text)

        assert "GPT-4" in description or "language model" in description.lower()

    def test_extract_description_not_found(self):
        """Test when entity is not mentioned in source."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        source_text = "This text doesn't mention the entity at all."

        description = service._extract_entity_description("SomeEntity", source_text)

        assert description == ""

    def test_extract_description_truncates_long_text(self):
        """Test that long descriptions are truncated."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        # Create a very long sentence
        source_text = "GPT-4 is " + "a very advanced " * 50 + "model."

        description = service._extract_entity_description(
            "GPT-4", source_text, max_length=100
        )

        assert len(description) <= 103  # 100 + "..."


class TestEnrichGraphDocuments:
    """Tests for graph document enrichment."""

    def test_enrichment_adds_source_tracking(self, sample_graph_document):
        """Test that enrichment adds source tracking properties."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        service._enrich_graph_documents([sample_graph_document])

        # Check nodes have source_document
        for node in sample_graph_document.nodes:
            assert "source_document" in node.properties

    def test_enrichment_adds_confidence(self, sample_graph_document):
        """Test that enrichment adds confidence scores."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        service._enrich_graph_documents([sample_graph_document])

        for node in sample_graph_document.nodes:
            assert "confidence" in node.properties
            assert 0 <= node.properties["confidence"] <= 1

    def test_enrichment_preserves_existing_description(self, sample_graph_document):
        """Test that existing descriptions are preserved."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        # Node already has a description
        original_desc = sample_graph_document.nodes[0].properties.get("description")

        service._enrich_graph_documents([sample_graph_document])

        # Description should be preserved or enhanced, not overwritten
        if original_desc:
            assert sample_graph_document.nodes[0].properties["description"] is not None


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_build_graph_batch_empty_docs(self):
        """Test batch processing with empty document list."""
        mock_repo = MagicMock()
        service = GraphService(mock_repo)

        result = service.build_graph_batch([])

        assert result["chunks_processed"] == 0
        assert result["graph_documents"] == 0

    def test_build_graph_batch_handles_errors(self, sample_documents):
        """Test that batch processing continues after errors."""
        mock_repo = MagicMock()
        mock_repo.add_graph_documents_batch.return_value = {
            "total_nodes": 0,
            "total_relationships": 0,
        }

        with patch.object(GraphService, "_process_batch_with_retry") as mock_process:
            mock_process.side_effect = [Exception("Batch failed"), []]

            service = GraphService(mock_repo)
            # Override rate limiter
            service.rate_limiter = MagicMock()
            service.rate_limiter.acquire.return_value = 0

            result = service.build_graph_batch(sample_documents, batch_size=1)

            # Should have recorded the failed batch
            assert result["chunks_failed"] > 0

    def test_build_graph_concurrent_flushes_buffered_writes(self):
        """Concurrent mode should flush writes in larger groups, not one DB call per batch."""

        class MockGraphDoc:
            def __init__(self):
                self.nodes = []
                self.relationships = []
                self.source = None

        mock_repo = MagicMock()
        mock_repo.add_graph_documents_batch.return_value = {
            "total_nodes": 0,
            "total_relationships": 0,
        }

        docs = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]

        service = GraphService(mock_repo)
        service.rate_limiter = MagicMock()
        service.rate_limiter.acquire.return_value = 0

        with (
            patch("src.services.graph_service.config.BATCH_SIZE_CHUNKS", 1),
            patch("src.services.graph_service.config.GRAPH_WRITE_FLUSH_SIZE", 2),
            patch.object(
                GraphService,
                "_process_single_batch",
                side_effect=[[MockGraphDoc()], [MockGraphDoc()]],
            ),
        ):
            result = service.build_graph_concurrent(docs, max_workers=2)

        assert result["graph_documents"] == 2
        assert mock_repo.add_graph_documents_batch.call_count == 1


class TestQueryMethods:
    """Tests for query methods."""

    def test_query_graph_hybrid_delegates_to_repo(self):
        """Test that hybrid query delegates to repository."""
        mock_repo = MagicMock()
        mock_repo.query_hybrid.return_value = {
            "answer": "Test answer",
            "entities_found": [],
            "graph_context": [],
            "confidence": 0.8,
        }

        service = GraphService(mock_repo)
        result = service.query_graph_hybrid("What is GPT-4?")

        mock_repo.query_hybrid.assert_called_once_with("What is GPT-4?")
        assert result["answer"] == "Test answer"

    def test_query_graph_legacy_delegates_to_repo(self):
        """Test that legacy query delegates to repository."""
        mock_repo = MagicMock()
        mock_repo.query.return_value = "Legacy answer"

        service = GraphService(mock_repo)
        result = service.query_graph("What is GPT-4?")

        mock_repo.query.assert_called_once_with("What is GPT-4?")
        assert result == "Legacy answer"


class TestDatabasePreparation:
    """Tests for database preparation."""

    def test_prepare_database_creates_indexes(self):
        """Test that prepare_database calls create_indexes."""
        mock_repo = MagicMock()

        service = GraphService(mock_repo)
        service.prepare_database()

        mock_repo.create_indexes.assert_called_once()


class TestTransformerConfiguration:
    """Tests for LLM transformer configuration."""

    def test_transformer_uses_config_schema(self):
        """Test that transformer is configured with allowed nodes/relationships."""
        mock_repo = MagicMock()

        with (
            patch("src.services.graph_service.ChatOpenAI") as mock_llm,
            patch("src.services.graph_service.LLMGraphTransformer") as mock_transformer,
        ):
            mock_llm.return_value = MagicMock()
            mock_transformer.return_value = MagicMock()

            service = GraphService(mock_repo)
            _ = service.transformer

            # Check that transformer was created with correct parameters
            call_kwargs = mock_transformer.call_args[1]
            assert call_kwargs["node_properties"] is True
            assert call_kwargs["relationship_properties"] is True
