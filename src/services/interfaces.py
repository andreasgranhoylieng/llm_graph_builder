"""
Service Interfaces - Abstract base classes for dependency injection.
Follows DDD pattern with clear boundaries between layers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IFileRepository(ABC):
    """Interface for file system operations."""

    @abstractmethod
    def get_files(self, folder_path: str, extensions: List[str]) -> List[str]:
        """Get list of files with specific extensions in a folder."""
        pass

    @abstractmethod
    def get_files_recursive(self, folder_path: str, extensions: List[str]) -> List[str]:
        """Recursively get files with specific extensions."""
        pass

    @abstractmethod
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        pass


class INeo4jRepository(ABC):
    """Interface for Neo4j graph database operations."""

    # ==========================================================================
    # DOCUMENT INGESTION
    # ==========================================================================

    @abstractmethod
    def add_graph_documents(self, graph_documents: List, include_source: bool = True):
        """Add graph documents to the database."""
        pass

    @abstractmethod
    def add_graph_documents_batch(
        self, graph_documents: List, batch_size: int = 100, include_source: bool = True
    ) -> dict:
        """Add graph documents in batches with embedding generation."""
        pass

    # ==========================================================================
    # QUERYING
    # ==========================================================================

    @abstractmethod
    def query(self, question: str) -> str:
        """Query the knowledge graph using natural language (Cypher-based)."""
        pass

    @abstractmethod
    def query_hybrid(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Hybrid query combining vector search and graph traversal.

        Returns:
            {
                "answer": str,
                "entities_found": List[dict],
                "graph_context": List[dict],
                "confidence": float
            }
        """
        pass

    # ==========================================================================
    # VECTOR SEARCH
    # ==========================================================================

    @abstractmethod
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        node_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on entity embeddings.

        Args:
            query: Natural language query
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            node_labels: Optional filter by node labels

        Returns:
            List of matching entities with scores
        """
        pass

    @abstractmethod
    def vector_search_documents(
        self, query: str, top_k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search document chunks by vector similarity."""
        pass

    # ==========================================================================
    # GRAPH TRAVERSAL
    # ==========================================================================

    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID property."""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes up to a certain depth.

        Args:
            node_id: Starting node ID
            depth: How many hops to traverse
            direction: "in", "out", or "both"
            relationship_types: Optional filter by relationship types

        Returns:
            List of nodes with their relationship info
        """
        pass

    @abstractmethod
    def find_path(
        self, start_id: str, end_id: str, max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two nodes."""
        pass

    @abstractmethod
    def get_subgraph(self, center_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get a subgraph centered on a node.

        Returns:
            {
                "nodes": List[dict],
                "relationships": List[dict]
            }
        """
        pass

    # ==========================================================================
    # DATABASE MANAGEMENT
    # ==========================================================================

    @abstractmethod
    def create_indexes(self):
        """Create database indexes for performance."""
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        """Get database statistics."""
        pass

    @abstractmethod
    def refresh_schema(self):
        """Refresh the schema cache."""
        pass

    @abstractmethod
    def clear_database(self, confirm: bool = False) -> int:
        """Clear all data from the database."""
        pass

    @abstractmethod
    def execute_cypher(self, query: str, params: Optional[dict] = None) -> List[dict]:
        """Execute a raw Cypher query (for advanced use cases)."""
        pass

    @abstractmethod
    def parallel_bfs_from_seeds(
        self, seed_ids: List[str], max_depth: int = 3
    ) -> Dict[str, Any]:
        """Launch parallel BFS from multiple seed nodes, merge results."""
        pass

    @abstractmethod
    def graph_rerank(
        self,
        entities: List[Dict[str, Any]],
        query_entity_ids: List[str],
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Rerank entity results by graph proximity to query entities."""
        pass


class ITraversalService(ABC):
    """Interface for advanced graph traversal operations."""

    @abstractmethod
    def find_connections(self, entity_a: str, entity_b: str) -> Dict[str, Any]:
        """Find how two entities are connected."""
        pass

    @abstractmethod
    def get_entity_context(
        self, entity_id: str, context_depth: int = 2
    ) -> Dict[str, Any]:
        """Get full context around an entity for RAG."""
        pass

    @abstractmethod
    def find_similar_entities(
        self, entity_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find entities similar to the given one."""
        pass

    @abstractmethod
    def resolve_entity(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a name or ID to a valid entity node.
        Prioritizes: Exact ID -> Exact Name -> Fuzzy Name -> Vector Search.
        """
        pass

    @abstractmethod
    def get_multi_entity_context(
        self, entity_ids: List[str], max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get aggregated context from multiple entities via parallel BFS.
        Returns merged subgraph with bridge nodes identified.
        """
        pass
