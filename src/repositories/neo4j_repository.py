"""
Neo4j Repository - Handles all Neo4j database operations with bulk insert support.
"""

from typing import List
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from src.services.interfaces import INeo4jRepository
from src import config
from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

Note: Do not include any explanations or apologies in your response.
Do not respond to any questions that might ask you to confirm anything other than appearing as a Cypher query engine.
Do not include any text except the generated Cypher statement.

Important:
1. For entity matching, prefer using the `entity_embeddings` vector index if the exact ID is not known or if common abbreviations are used.
   Example for vector search:
   MATCH (n:__Entity__)
   CALL db.index.vector.queryNodes('entity_embeddings', 1, $embedding) YIELD node, score
   WHERE score > 0.8
   RETURN node

2. Alternatively, use case-insensitive matching with `toLower()` and `CONTAINS` for robust string lookups.
   Example: MATCH (n:Organization) WHERE toLower(n.id) CONTAINS toLower('xai') RETURN n

3. Always return relevant nodes and their relationships to answer the question.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


class Neo4jRepository(INeo4jRepository):
    """Repository for Neo4j graph database operations with batch support."""

    def __init__(self):
        self._graph = None
        self._embeddings = None

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

                # --- EMBEDDING GENERATION ---
                # 1. Extract unique entities from this batch
                nodes_to_embed = {}  # id -> label
                for g_doc in batch:
                    for node in g_doc.nodes:
                        if node.id not in nodes_to_embed:
                            nodes_to_embed[node.id] = node.type

                if nodes_to_embed:
                    node_ids = list(nodes_to_embed.keys())
                    embeddings = self._get_embeddings().embed_documents(node_ids)

                    update_query = """
                    UNWIND $data as item
                    MATCH (n {id: item.id})
                    SET n.embedding = item.embedding
                    """
                    data = [
                        {"id": nid, "embedding": emb}
                        for nid, emb in zip(node_ids, embeddings)
                    ]
                    graph.query(update_query, params={"data": data})

                # 2. Extract Document chunks to embed
                docs_to_embed = []
                for g_doc in batch:
                    if hasattr(g_doc, "source"):
                        doc_id = g_doc.source.metadata.get(
                            "id"
                        ) or g_doc.source.metadata.get("source")
                        doc_content = g_doc.source.page_content
                        if doc_id and doc_content:
                            docs_to_embed.append({"id": doc_id, "content": doc_content})

                if docs_to_embed:
                    doc_contents = [d["content"] for d in docs_to_embed]
                    doc_embeddings = self._get_embeddings().embed_documents(
                        doc_contents
                    )

                    doc_update_query = """
                    UNWIND $data as item
                    MATCH (d:Document {id: item.id})
                    SET d.embedding = item.embedding
                    """
                    doc_data = [
                        {"id": d["id"], "embedding": emb}
                        for d, emb in zip(docs_to_embed, doc_embeddings)
                    ]
                    graph.query(doc_update_query, params={"data": doc_data})
                # ----------------------------

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
            # Vector Index for entities
            f"CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS FOR (n:__Entity__) ON (n.embedding) OPTIONS {{indexConfig: {{ `vector.dimensions`: {config.EMBEDDING_DIMENSION}, `vector.similarity_function`: 'cosine' }}}},",
            # Vector Index for documents (chunks)
            f"CREATE VECTOR INDEX document_embeddings IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS {{indexConfig: {{ `vector.dimensions`: {config.EMBEDDING_DIMENSION}, `vector.similarity_function`: 'cosine' }}}}",
        ]

        for query in index_queries:
            try:
                graph.query(query)
            except Exception:
                # Index might already exist or not be supported
                pass

        print("Database indexes created/verified")

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

        # We can pass the embeddings to the chain if needed for vector search in Cypher
        # However, GraphCypherQAChain usually handles direct query generation.
        # To support vector search, we might need to pre-process the question or
        # use a custom chain. For now, we'll use the prompt to guide the LLM.

        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

        try:
            # We provide the embedding of the question in case the Cypher query wants to use it
            question_embedding = self._get_embeddings().embed_query(question)

            response = chain.invoke(
                {"query": question, "embedding": question_embedding}
            )
            return response.get("result", "No answer found.")
        except Exception as e:
            return f"Error processing query: {e}"

    def refresh_schema(self):
        self._get_graph().refresh_schema()

    def _get_embeddings(self):
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY
            )
        return self._embeddings
