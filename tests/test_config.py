"""
Tests for configuration schema and validation.
"""

from src import config


class TestNodeSchema:
    """Tests for node type configuration."""

    def test_allowed_nodes_includes_ai_types(self):
        """Test that AI-specific node types are defined."""
        ai_types = [
            "AIModel",
            "AICompany",
            "Researcher",
            "Paper",
            "Architecture",
            "Technique",
        ]

        for node_type in ai_types:
            assert node_type in config.ALLOWED_NODES, f"Missing node type: {node_type}"

    def test_allowed_nodes_non_empty(self):
        """Test that node list is not empty."""
        assert len(config.ALLOWED_NODES) > 0

    def test_no_duplicate_node_types(self):
        """Test that there are no duplicate node types."""
        assert len(config.ALLOWED_NODES) == len(set(config.ALLOWED_NODES))


class TestRelationshipSchema:
    """Tests for relationship type configuration."""

    def test_allowed_relationships_includes_core_types(self):
        """Test that core relationship types are defined."""
        core_rels = [
            "DEVELOPED_BY",
            "USES_ARCHITECTURE",
            "IMPLEMENTS",
            "TRAINED_ON",
            "EVALUATED_ON",
        ]

        for rel_type in core_rels:
            assert rel_type in config.ALLOWED_RELATIONSHIPS, (
                f"Missing relationship: {rel_type}"
            )

    def test_allowed_relationships_non_empty(self):
        """Test that relationship list is not empty."""
        assert len(config.ALLOWED_RELATIONSHIPS) > 0

    def test_no_duplicate_relationship_types(self):
        """Test that there are no duplicate relationship types."""
        assert len(config.ALLOWED_RELATIONSHIPS) == len(
            set(config.ALLOWED_RELATIONSHIPS)
        )


class TestPropertySchema:
    """Tests for node property schema configuration."""

    def test_base_entity_schema_exists(self):
        """Test that base entity schema is defined."""
        assert "__Entity__" in config.NODE_PROPERTY_SCHEMA

    def test_ai_model_schema_exists(self):
        """Test that AIModel schema is defined with AI-specific properties."""
        assert "AIModel" in config.NODE_PROPERTY_SCHEMA

        ai_model_schema = config.NODE_PROPERTY_SCHEMA["AIModel"]
        assert "required" in ai_model_schema
        assert "optional" in ai_model_schema
        assert "embedding_fields" in ai_model_schema

    def test_embedding_fields_defined(self):
        """Test that embedding fields are defined for all schema types."""
        for node_type, schema in config.NODE_PROPERTY_SCHEMA.items():
            assert "embedding_fields" in schema, (
                f"Missing embedding_fields for {node_type}"
            )
            assert len(schema["embedding_fields"]) > 0, (
                f"Empty embedding_fields for {node_type}"
            )


class TestEmbeddingConfiguration:
    """Tests for embedding configuration."""

    def test_embedding_dimension_matches_model(self):
        """Test that embedding dimension matches the model."""
        # text-embedding-3-large has 3072 dimensions
        assert config.EMBEDDING_DIMENSION == 3072

    def test_embedding_model_defined(self):
        """Test that embedding model is defined."""
        assert config.EMBEDDING_MODEL is not None
        assert len(config.EMBEDDING_MODEL) > 0


class TestVectorSearchConfiguration:
    """Tests for vector search configuration."""

    def test_vector_search_defaults(self):
        """Test that vector search defaults are set."""
        assert config.VECTOR_SEARCH_TOP_K > 0
        assert 0 <= config.VECTOR_SEARCH_SCORE_THRESHOLD <= 1
        assert config.HYBRID_SEARCH_DEPTH > 0


class TestRateLimitConfiguration:
    """Tests for rate limit configuration."""

    def test_rate_limits_are_positive(self):
        """Test that rate limits are positive values."""
        assert config.RATE_LIMIT_RPM > 0
        assert config.RATE_LIMIT_TPM > 0


class TestProcessingConfiguration:
    """Tests for batch processing configuration."""

    def test_batch_sizes_are_positive(self):
        """Test that batch sizes are positive."""
        assert config.BATCH_SIZE_FILES > 0
        assert config.BATCH_SIZE_CHUNKS > 0

    def test_retry_configuration(self):
        """Test retry configuration is valid."""
        assert config.MAX_RETRIES > 0
        assert config.RETRY_DELAY_SECONDS > 0
