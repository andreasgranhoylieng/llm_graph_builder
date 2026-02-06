"""
Neo4j Repository - Handles all Neo4j database operations with bulk insert support.
"""

from typing import List
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from src.services.interfaces import INeo4jRepository
from src import config


class Neo4jRepository(INeo4jRepository):
    """Repository for Neo4j graph database operations with batch support."""

    def __init__(self):
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            try:
                self._graph = Neo4jGraph(
                    url=config.NEO4J_URI,
                    username=config.NEO4J_USERNAME,
                    password=config.NEO4J_PASSWORD,
                    database=config.NEO4J_DATABASE,
                )
            except Exception as e:
                # Add more diagnostic info
                error_msg = str(e)
                if "Unable to retrieve routing information" in error_msg:
                    raise ConnectionError(
                        f"Could not connect to Neo4j at {config.NEO4J_URI}. "
                        "This often means the database is paused, offline, or the URL is incorrect. "
                        f"Original error: {error_msg}"
                    )
                raise ConnectionError(f"Neo4j connection failed: {error_msg}")
        return self._graph

    def add_graph_documents(self, graph_documents: List, include_source: bool = True):
        """Add graph documents to Neo4j."""
        graph = self._get_graph()
        graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=include_source
        )

    def add_graph_documents_batch(
        self, graph_documents: List, batch_size: int = 100, include_source: bool = True
    ) -> dict:
        """
        Insert graph documents in batches to avoid transaction timeouts.

        Args:
            graph_documents: List of graph documents to insert
            batch_size: Number of documents per batch
            include_source: Whether to include source document reference

        Returns:
            Statistics about the insertion
        """
        graph = self._get_graph()

        total_nodes = 0
        total_relationships = 0
        batches_processed = 0

        for i in range(0, len(graph_documents), batch_size):
            batch = graph_documents[i : i + batch_size]

            try:
                graph.add_graph_documents(
                    batch, baseEntityLabel=True, include_source=include_source
                )

                # Count nodes and relationships
                for doc in batch:
                    total_nodes += len(doc.nodes) if hasattr(doc, "nodes") else 0
                    total_relationships += (
                        len(doc.relationships) if hasattr(doc, "relationships") else 0
                    )

                batches_processed += 1

            except Exception as e:
                print(f"⚠️ Batch {batches_processed} failed: {e}")
                # Continue with next batch
                continue

        return {
            "batches_processed": batches_processed,
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
        }

    def create_indexes(self):
        """
        Create indexes for better query performance.
        Should be called before large-scale ingestion.
        """
        graph = self._get_graph()

        index_queries = [
            # Index on common node labels
            "CREATE INDEX IF NOT EXISTS FOR (n:Person) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Organization) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Technology) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Event) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Product) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Location) ON (n.id)",
            # Index on Document source
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
            # Full-text search index (optional)
            # "CREATE FULLTEXT INDEX entityNames IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.id]",
        ]

        for query in index_queries:
            try:
                graph.query(query)
            except Exception:
                # Index might already exist or not be supported
                pass

        print("📇 Database indexes created/verified")

    def get_statistics(self) -> dict:
        """Get database statistics."""
        graph = self._get_graph()

        try:
            # Count nodes
            node_count_result = graph.query("MATCH (n) RETURN count(n) as count")
            node_count = node_count_result[0]["count"] if node_count_result else 0

            # Count relationships
            rel_count_result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_count_result[0]["count"] if rel_count_result else 0

            # Count by label
            label_counts_result = graph.query(
                "MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC LIMIT 10"
            )

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "label_counts": label_counts_result or [],
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_database(self, confirm: bool = False):
        """
        Clear all nodes and relationships from the database.
        Requires explicit confirmation.
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear database")

        graph = self._get_graph()

        # Delete in batches to avoid memory issues
        batch_size = 10000
        deleted_total = 0

        while True:
            result = graph.query(
                f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(*) as deleted"
            )
            deleted = result[0]["deleted"] if result else 0
            deleted_total += deleted

            if deleted < batch_size:
                break

        print(f"🗑️ Cleared {deleted_total} nodes from database")
        return deleted_total

    def query(self, question: str) -> str:
        """Query the knowledge graph using natural language."""
        graph = self._get_graph()
        graph.refresh_schema()

        llm = ChatOpenAI(temperature=0, model_name=config.LLM_MODEL)

        chain = GraphCypherQAChain.from_llm(
            graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
        )

        try:
            response = chain.invoke({"query": question})
            return response.get("result", "No answer found.")
        except Exception as e:
            return f"Error processing query: {e}"

    def refresh_schema(self):
        self._get_graph().refresh_schema()
