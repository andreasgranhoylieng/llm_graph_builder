"""
Neo4j Repository - Comprehensive graph database operations with vector search and traversal.
"""

from typing import List, Dict, Any, Optional
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from src.services.interfaces import INeo4jRepository
from src import config


# =============================================================================
# CYPHER GENERATION PROMPT
# =============================================================================
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database about Generative AI.

Schema:
{schema}

Important Query Guidelines:
1. For entity matching, use case-insensitive matching with toLower() and CONTAINS for fuzzy text matching.
   Example: WHERE toLower(n.name) CONTAINS toLower('gpt')

2. Use the following node types for AI knowledge graphs:
   - AIModel: AI models like GPT-4, Claude, Llama
   - AICompany: Companies like OpenAI, Anthropic, Google DeepMind
   - Researcher: Key AI researchers
   - Paper: Research papers
   - Architecture: Model architectures like Transformer
   - Technique: Methods like RLHF, RAG, LoRA
   - Benchmark: Evaluation benchmarks like MMLU, HumanEval

3. Important relationships:
   - DEVELOPED_BY: Model to Company/Researcher
   - USES_ARCHITECTURE: Model to Architecture
   - IMPLEMENTS: Model to Technique
   - TRAINED_ON: Model to Dataset
   - FINE_TUNED_FROM: Model to base Model
   - EVALUATED_ON / ACHIEVES: Model to Benchmark

4. Always return relevant properties: name, description, and relationship context.

5. For multi-hop queries, use variable-length paths: (a)-[*1..3]-(b)

Rules:
- Do NOT include any explanations or apologies
- Do NOT respond to anything except generating Cypher
- Return ONLY the Cypher statement

Question: {question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


class Neo4jRepository(INeo4jRepository):
    """Repository for Neo4j graph database operations with vector search and traversal."""

    def __init__(self):
        self._graph: Optional[Neo4jGraph] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

    # =========================================================================
    # CONNECTION MANAGEMENT
    # =========================================================================

    def _get_graph(self) -> Neo4jGraph:
        """Get or create Neo4j connection with error handling."""
        if self._graph is None:
            try:
                self._graph = Neo4jGraph(
                    url=config.NEO4J_URI,
                    username=config.NEO4J_USERNAME,
                    password=config.NEO4J_PASSWORD,
                    database=config.NEO4J_DATABASE,
                )
            except Exception as e:
                error_msg = str(e)
                if "Unable to retrieve routing information" in error_msg:
                    raise ConnectionError(
                        f"Could not connect to Neo4j at {config.NEO4J_URI}. "
                        "The database may be paused, offline, or the URL is incorrect. "
                        f"Original error: {error_msg}"
                    )
                raise ConnectionError(f"Neo4j connection failed: {error_msg}")
        return self._graph

    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL, openai_api_key=config.OPENAI_API_KEY
            )
        return self._embeddings

    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================

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
        Insert graph documents in batches with embedding generation.

        Generates embeddings for:
        1. Entity nodes (using name + description)
        2. Document chunks (using full content)
        """
        graph = self._get_graph()

        total_nodes = 0
        total_relationships = 0
        batches_processed = 0

        for i in range(0, len(graph_documents), batch_size):
            batch = graph_documents[i : i + batch_size]

            try:
                # Insert graph documents
                graph.add_graph_documents(
                    batch, baseEntityLabel=True, include_source=include_source
                )

                # --- ENTITY EMBEDDING GENERATION ---
                nodes_to_embed = {}  # id -> {type, description}
                for g_doc in batch:
                    for node in g_doc.nodes:
                        if node.id not in nodes_to_embed:
                            # Extract description from properties if available
                            description = ""
                            if hasattr(node, "properties") and node.properties:
                                description = node.properties.get("description", "")
                            nodes_to_embed[node.id] = {
                                "type": node.type,
                                "description": description,
                            }

                if nodes_to_embed:
                    # Create embedding text: "name: description" for better semantic matching
                    node_ids = list(nodes_to_embed.keys())
                    texts_to_embed = []
                    for nid in node_ids:
                        desc = nodes_to_embed[nid]["description"]
                        if desc:
                            texts_to_embed.append(f"{nid}: {desc}")
                        else:
                            texts_to_embed.append(nid)

                    embeddings = self._get_embeddings().embed_documents(texts_to_embed)

                    update_query = """
                    UNWIND $data as item
                    MATCH (n:__Entity__ {id: item.id})
                    SET n.embedding = item.embedding,
                        n.embedding_text = item.embedding_text,
                        n.embedded_at = datetime()
                    """
                    data = [
                        {"id": nid, "embedding": emb, "embedding_text": txt}
                        for nid, emb, txt in zip(node_ids, embeddings, texts_to_embed)
                    ]
                    graph.query(update_query, params={"data": data})

                # --- DOCUMENT CHUNK EMBEDDING ---
                docs_to_embed = []
                for g_doc in batch:
                    if hasattr(g_doc, "source") and g_doc.source:
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
                    SET d.embedding = item.embedding,
                        d.embedded_at = datetime()
                    """
                    doc_data = [
                        {"id": d["id"], "embedding": emb}
                        for d, emb in zip(docs_to_embed, doc_embeddings)
                    ]
                    graph.query(doc_update_query, params={"data": doc_data})

                # Count nodes and relationships
                for doc in batch:
                    total_nodes += len(doc.nodes) if hasattr(doc, "nodes") else 0
                    total_relationships += (
                        len(doc.relationships) if hasattr(doc, "relationships") else 0
                    )

                batches_processed += 1

            except Exception as e:
                print(f"⚠️ Batch {batches_processed + 1} failed: {e}")
                continue

        return {
            "batches_processed": batches_processed,
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
        }

    # =========================================================================
    # VECTOR SEARCH
    # =========================================================================

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        node_labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on entity embeddings.
        """
        graph = self._get_graph()
        query_embedding = self._get_embeddings().embed_query(query)

        # Build label filter if specified
        label_filter = ""
        if node_labels:
            labels_str = " OR ".join(
                [f"'{label}' IN labels(node)" for label in node_labels]
            )
            label_filter = f"WHERE ({labels_str})"

        cypher = f"""
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        {label_filter}
        WHERE score >= $threshold
        RETURN 
            node.id as id,
            node.name as name,
            node.description as description,
            labels(node) as labels,
            score
        ORDER BY score DESC
        """

        try:
            results = graph.query(
                cypher,
                params={
                    "embedding": query_embedding,
                    "top_k": top_k,
                    "threshold": score_threshold,
                },
            )
            return results or []
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []

    def vector_search_documents(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search document chunks by vector similarity."""
        graph = self._get_graph()
        query_embedding = self._get_embeddings().embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('document_embeddings', $top_k, $embedding)
        YIELD node, score
        RETURN 
            node.id as id,
            node.text as content,
            score
        ORDER BY score DESC
        """

        try:
            results = graph.query(
                cypher, params={"embedding": query_embedding, "top_k": top_k}
            )
            return results or []
        except Exception as e:
            print(f"⚠️ Document vector search failed: {e}")
            return []

    # =========================================================================
    # GRAPH TRAVERSAL
    # =========================================================================

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID property."""
        graph = self._get_graph()

        cypher = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        LIMIT 1
        """

        results = graph.query(cypher, params={"node_id": node_id})
        if results:
            node_data = dict(results[0]["n"])
            node_data["labels"] = results[0]["labels"]
            return node_data
        return None

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes up to a certain depth."""
        graph = self._get_graph()

        # Build relationship pattern
        rel_pattern = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_pattern = f":{rel_types}"

        # Direction pattern
        if direction == "out":
            path_pattern = f"-[r{rel_pattern}*1..{depth}]->"
        elif direction == "in":
            path_pattern = f"<-[r{rel_pattern}*1..{depth}]-"
        else:
            path_pattern = f"-[r{rel_pattern}*1..{depth}]-"

        cypher = f"""
        MATCH (start {{id: $node_id}}){path_pattern}(neighbor)
        WHERE start <> neighbor
        WITH DISTINCT neighbor, 
             [(start){path_pattern}(neighbor) | type(head(r))] as rel_types
        RETURN 
            neighbor.id as id,
            neighbor.name as name,
            neighbor.description as description,
            labels(neighbor) as labels,
            rel_types[0] as relationship
        LIMIT 50
        """

        try:
            results = graph.query(cypher, params={"node_id": node_id})
            return results or []
        except Exception as e:
            print(f"⚠️ Neighbor search failed: {e}")
            return []

    def find_path(
        self, start_id: str, end_id: str, max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two nodes."""
        graph = self._get_graph()

        cypher = f"""
        MATCH path = shortestPath(
            (start {{id: $start_id}})-[*1..{max_depth}]-(end {{id: $end_id}})
        )
        RETURN 
            [n IN nodes(path) | {{id: n.id, name: n.name, labels: labels(n)}}] as nodes,
            [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] as relationships
        """

        try:
            results = graph.query(
                cypher, params={"start_id": start_id, "end_id": end_id}
            )
            if results:
                return {
                    "nodes": results[0]["nodes"],
                    "relationships": results[0]["relationships"],
                }
            return None
        except Exception as e:
            print(f"⚠️ Path finding failed: {e}")
            return None

    def get_subgraph(self, center_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a subgraph centered on a node."""
        graph = self._get_graph()

        cypher = f"""
        MATCH (center {{id: $center_id}})
        CALL apoc.path.subgraphAll(center, {{maxLevel: {max_depth}}})
        YIELD nodes, relationships
        RETURN 
            [n IN nodes | {{
                id: n.id, 
                name: n.name, 
                labels: labels(n),
                description: n.description
            }}] as nodes,
            [r IN relationships | {{
                type: type(r),
                start: startNode(r).id,
                end: endNode(r).id,
                properties: properties(r)
            }}] as relationships
        """

        try:
            results = graph.query(cypher, params={"center_id": center_id})
            if results:
                return {
                    "nodes": results[0]["nodes"],
                    "relationships": results[0]["relationships"],
                }
        except Exception:
            # APOC might not be installed, fallback to basic query
            pass

        # Fallback without APOC
        fallback_cypher = f"""
        MATCH (center {{id: $center_id}})-[r*0..{max_depth}]-(connected)
        WITH DISTINCT connected, r
        RETURN 
            collect(DISTINCT {{
                id: connected.id,
                name: connected.name,
                labels: labels(connected),
                description: connected.description
            }}) as nodes
        """

        results = graph.query(fallback_cypher, params={"center_id": center_id})
        return {"nodes": results[0]["nodes"] if results else [], "relationships": []}

    # =========================================================================
    # HYBRID QUERY
    # =========================================================================

    def query_hybrid(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Hybrid query combining vector search and graph traversal.

        Pipeline:
        1. Vector search to find relevant entities
        2. Expand graph context around found entities
        3. Use LLM to synthesize answer from context
        """
        # Step 1: Vector search for relevant entities
        entities = self.vector_search(
            question, top_k=top_k, score_threshold=config.VECTOR_SEARCH_SCORE_THRESHOLD
        )

        if not entities:
            # Fallback to pure Cypher query
            return {
                "answer": self.query(question),
                "entities_found": [],
                "graph_context": [],
                "confidence": 0.3,
                "method": "cypher_fallback",
            }

        # Step 2: Expand graph context around top entities
        graph_context = []
        for entity in entities[:3]:  # Top 3 entities
            neighbors = self.get_neighbors(
                entity["id"], depth=config.HYBRID_SEARCH_DEPTH
            )
            graph_context.extend(neighbors)

        # Step 3: Build context for LLM
        context_parts = []

        # Add entity information
        context_parts.append("## Relevant Entities Found:")
        for ent in entities[:5]:
            labels = ", ".join(ent.get("labels", []))
            desc = ent.get("description", "No description")
            context_parts.append(f"- **{ent['name']}** ({labels}): {desc}")

        # Add relationship context
        if graph_context:
            context_parts.append("\n## Related Entities:")
            seen = set()
            for ctx in graph_context[:10]:
                if ctx["id"] not in seen:
                    seen.add(ctx["id"])
                    rel = ctx.get("relationship", "RELATED_TO")
                    context_parts.append(f"- {ctx['name']} ({rel})")

        context_str = "\n".join(context_parts)

        # Step 4: Use LLM to synthesize answer
        llm = ChatOpenAI(temperature=0, model_name=config.LLM_MODEL)

        prompt = f"""Based on the following knowledge graph context, answer the question.

{context_str}

Question: {question}

Provide a clear, factual answer based only on the information above. If the information is insufficient, say so.

Answer:"""

        try:
            response = llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "entities_found": entities,
            "graph_context": graph_context[:10],
            "confidence": min(entities[0]["score"], 1.0) if entities else 0.0,
            "method": "hybrid",
        }

    def query(self, question: str) -> str:
        """Query the knowledge graph using natural language (Cypher-based)."""
        graph = self._get_graph()
        graph.refresh_schema()

        llm = ChatOpenAI(temperature=0, model_name=config.LLM_MODEL)

        chain = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=llm,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

        try:
            response = chain.invoke({"query": question})
            return response.get("result", "No answer found.")
        except Exception as e:
            return f"Error processing query: {e}"

    # =========================================================================
    # DATABASE MANAGEMENT
    # =========================================================================

    def create_indexes(self):
        """Create indexes for better query performance."""
        graph = self._get_graph()

        index_queries = [
            # Indexes for AI entity types
            "CREATE INDEX IF NOT EXISTS FOR (n:AIModel) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:AICompany) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Researcher) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Paper) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Architecture) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Technique) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Benchmark) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Framework) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Application) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Dataset) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id)",
            # Full-text search index for names
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:__Entity__) ON EACH [n.name, n.description]",
            # Vector indexes
            f"CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS FOR (n:__Entity__) ON (n.embedding) OPTIONS {{indexConfig: {{ `vector.dimensions`: {config.EMBEDDING_DIMENSION}, `vector.similarity_function`: 'cosine' }}}}",
            f"CREATE VECTOR INDEX document_embeddings IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS {{indexConfig: {{ `vector.dimensions`: {config.EMBEDDING_DIMENSION}, `vector.similarity_function`: 'cosine' }}}}",
        ]

        for query in index_queries:
            try:
                graph.query(query)
            except Exception as e:
                # Index might already exist or syntax not supported
                if "already exists" not in str(e).lower():
                    print(f"⚠️ Index creation note: {e}")

        print("✅ Database indexes created/verified")

    def get_statistics(self) -> dict:
        """Get database statistics."""
        graph = self._get_graph()

        try:
            node_count_result = graph.query("MATCH (n) RETURN count(n) as count")
            node_count = node_count_result[0]["count"] if node_count_result else 0

            rel_count_result = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_count_result[0]["count"] if rel_count_result else 0

            label_counts_result = graph.query(
                "MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC LIMIT 15"
            )

            # Count entities with embeddings
            embedded_count_result = graph.query(
                "MATCH (n:__Entity__) WHERE n.embedding IS NOT NULL RETURN count(n) as count"
            )
            embedded_count = (
                embedded_count_result[0]["count"] if embedded_count_result else 0
            )

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "nodes_with_embeddings": embedded_count,
                "label_counts": label_counts_result or [],
            }
        except Exception as e:
            return {"error": str(e)}

    def refresh_schema(self):
        """Refresh the schema cache."""
        self._get_graph().refresh_schema()

    def clear_database(self, confirm: bool = False) -> int:
        """Clear all nodes and relationships from the database."""
        if not confirm:
            raise ValueError("Must set confirm=True to clear database")

        graph = self._get_graph()
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

    def execute_cypher(self, query: str, params: Optional[dict] = None) -> List[dict]:
        """Execute a raw Cypher query."""
        graph = self._get_graph()
        return graph.query(query, params=params or {})
