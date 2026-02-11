"""
Neo4j Repository - Comprehensive graph database operations with vector search and traversal.
Enhanced with SOTA GraphRAG: bidirectional BFS, parallel multi-source BFS, graph-aware reranking.
"""

import hashlib
import json
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self._agent_llms: Dict[str, ChatOpenAI] = {}

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

    def _get_agent_llm(self, agent_name: str) -> ChatOpenAI:
        """Lazy initialization for role-specific LLM agents."""
        if agent_name not in self._agent_llms:
            self._agent_llms[agent_name] = ChatOpenAI(
                temperature=0, model_name=config.LLM_MODEL
            )
        return self._agent_llms[agent_name]

    # =========================================================================
    # DOCUMENT INGESTION
    # =========================================================================

    def add_graph_documents(self, graph_documents: List, include_source: bool = True):
        """Add graph documents to Neo4j."""
        graph = self._get_graph()
        graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=include_source
        )

    def verify_connectivity(self) -> bool:
        """Verify Neo4j connection is active, recombining if necessary."""
        try:
            if self._graph:
                self._graph.query("RETURN 1")
                return True
        except Exception:
            # Force reconnection
            self._graph = None

        try:
            self._get_graph()
            return True
        except Exception as e:
            print(f"⚠️ Connection verification failed: {e}")
            return False

    def add_graph_documents_batch(
        self, graph_documents: List, batch_size: int = 100, include_source: bool = True
    ) -> dict:
        """
        Insert graph documents in batches with embedding generation.

        Generates embeddings for:
        1. Entity nodes (using name + description)
        2. Document chunks (using full content)
        """
        self.verify_connectivity()
        graph = self._get_graph()

        batch_size = max(1, batch_size)
        embedding_batch_size = max(1, config.EMBEDDING_BATCH_SIZE)

        total_nodes = 0
        total_relationships = 0
        batches_processed = 0

        embedded_node_ids: Set[str] = set()
        embedded_doc_ids: Set[str] = set()

        for i in range(0, len(graph_documents), batch_size):
            batch = graph_documents[i : i + batch_size]

            try:
                graph.add_graph_documents(
                    batch, baseEntityLabel=True, include_source=include_source
                )

                nodes_to_embed = self._collect_nodes_for_embedding(
                    batch, embedded_node_ids
                )
                docs_to_embed = self._collect_documents_for_embedding(
                    batch, embedded_doc_ids
                )

                if config.SKIP_EXISTING_EMBEDDINGS and nodes_to_embed:
                    missing_node_ids = self._filter_ids_without_embedding(
                        graph, list(nodes_to_embed.keys()), label="__Entity__"
                    )
                    nodes_to_embed = {
                        node_id: nodes_to_embed[node_id]
                        for node_id in missing_node_ids
                        if node_id in nodes_to_embed
                    }

                if config.SKIP_EXISTING_EMBEDDINGS and docs_to_embed:
                    missing_doc_ids = self._filter_ids_without_embedding(
                        graph, list(docs_to_embed.keys()), label="Document"
                    )
                    docs_to_embed = {
                        doc_id: docs_to_embed[doc_id]
                        for doc_id in missing_doc_ids
                        if doc_id in docs_to_embed
                    }

                if nodes_to_embed:
                    node_ids = list(nodes_to_embed.keys())
                    node_texts = [
                        (
                            f"{nodes_to_embed[node_id]['name']}: {nodes_to_embed[node_id]['description']}"
                            if nodes_to_embed[node_id]["description"]
                            else nodes_to_embed[node_id]["name"]
                        )
                        for node_id in node_ids
                    ]
                    node_embeddings = self._embed_texts_in_batches(
                        node_texts, batch_size=embedding_batch_size
                    )

                    node_data = [
                        {"id": nid, "embedding": emb, "embedding_text": txt}
                        for nid, emb, txt in zip(node_ids, node_embeddings, node_texts)
                    ]
                    graph.query(
                        """
                        UNWIND $data as item
                        MATCH (n:__Entity__ {id: item.id})
                        SET n.embedding = item.embedding,
                            n.embedding_text = item.embedding_text,
                            n.embedded_at = datetime()
                        """,
                        params={"data": node_data},
                    )

                if docs_to_embed:
                    doc_ids = list(docs_to_embed.keys())
                    doc_texts = [docs_to_embed[doc_id] for doc_id in doc_ids]
                    doc_embeddings = self._embed_texts_in_batches(
                        doc_texts, batch_size=embedding_batch_size
                    )

                    doc_data = [
                        {"id": doc_id, "embedding": emb}
                        for doc_id, emb in zip(doc_ids, doc_embeddings)
                    ]
                    graph.query(
                        """
                        UNWIND $data as item
                        MATCH (d:Document {id: item.id})
                        SET d.embedding = item.embedding,
                            d.embedded_at = datetime()
                        """,
                        params={"data": doc_data},
                    )

                for doc in batch:
                    total_nodes += len(doc.nodes) if hasattr(doc, "nodes") else 0
                    total_relationships += (
                        len(doc.relationships) if hasattr(doc, "relationships") else 0
                    )

                batches_processed += 1

            except Exception as e:
                print(f"Batch {batches_processed + 1} failed: {e}")
                continue

        return {
            "batches_processed": batches_processed,
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
        }

    def _collect_nodes_for_embedding(
        self, graph_documents: List, seen_node_ids: Set[str]
    ) -> Dict[str, Dict[str, str]]:
        """Collect unique node texts for embedding from a batch."""
        nodes_to_embed: Dict[str, Dict[str, str]] = {}

        for g_doc in graph_documents:
            for node in getattr(g_doc, "nodes", []):
                node_id = str(getattr(node, "id", "") or "")
                if not node_id or node_id in seen_node_ids:
                    continue

                seen_node_ids.add(node_id)
                name = node_id
                description = ""
                properties = getattr(node, "properties", {}) or {}
                if properties:
                    name = properties.get("name") or name
                    description = properties.get("description", "")

                nodes_to_embed[node_id] = {
                    "name": str(name),
                    "description": str(description or ""),
                }

        return nodes_to_embed

    def _collect_documents_for_embedding(
        self, graph_documents: List, seen_doc_ids: Set[str]
    ) -> Dict[str, str]:
        """Collect unique document chunks for embedding from a batch."""
        docs_to_embed: Dict[str, str] = {}

        for g_doc in graph_documents:
            source = getattr(g_doc, "source", None)
            if not source:
                continue

            metadata = (getattr(source, "metadata", None) or {}).copy()
            content = getattr(source, "page_content", None)
            if not content:
                continue

            source_name = metadata.get("source") or metadata.get("source_file", "unknown")
            page = metadata.get("page", 0)
            chunk_id = metadata.get("chunk_id") or metadata.get("id")
            if not chunk_id:
                content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
                chunk_id = f"{source_name}#p{page}:{content_hash}"

            chunk_id = str(chunk_id)
            if chunk_id in seen_doc_ids:
                continue

            seen_doc_ids.add(chunk_id)
            docs_to_embed[chunk_id] = str(content)

        return docs_to_embed

    def _embed_texts_in_batches(
        self, texts: List[str], batch_size: int
    ) -> List[List[float]]:
        """Embed text in API-sized slices to improve stability and throughput."""
        if not texts:
            return []

        embeddings_model = self._get_embeddings()
        vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            text_batch = texts[i : i + batch_size]
            vectors.extend(embeddings_model.embed_documents(text_batch))

        return vectors

    def _filter_ids_without_embedding(
        self, graph: Neo4jGraph, ids: List[str], label: str
    ) -> List[str]:
        """
        Return ids that do not yet have an embedding.
        Falls back to the original ids when query responses are unavailable/invalid.
        """
        if not ids:
            return []

        ids = [str(item) for item in ids if item]
        if not ids:
            return []

        if label not in {"__Entity__", "Document"}:
            return ids

        result_ids: List[str] = []
        query = (
            """
            UNWIND $ids as id
            MATCH (n:__Entity__ {id: id})
            WHERE n.embedding IS NULL
            RETURN n.id as id
            """
            if label == "__Entity__"
            else """
            UNWIND $ids as id
            MATCH (d:Document {id: id})
            WHERE d.embedding IS NULL
            RETURN d.id as id
            """
        )

        for i in range(0, len(ids), 1000):
            id_batch = ids[i : i + 1000]
            try:
                rows = graph.query(query, params={"ids": id_batch})
                if not isinstance(rows, list):
                    return ids
                result_ids.extend(
                    str(row.get("id"))
                    for row in rows
                    if isinstance(row, dict) and row.get("id")
                )
            except Exception:
                return ids

        return result_ids

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

        cypher = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $embedding)
        YIELD node, score
        WHERE score >= $threshold
          AND ($node_labels IS NULL OR size($node_labels) = 0 OR ANY(label IN labels(node) WHERE label IN $node_labels))
        RETURN 
            node.id as id,
            COALESCE(node.name, node.id) as name,
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
                    "node_labels": node_labels or [],
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
        WITH node, score, [k IN ['text', 'page_content', 'content'] WHERE k IN keys(node)] as content_keys
        RETURN 
            node.id as id,
            CASE
                WHEN size(content_keys) > 0 THEN toString(node[content_keys[0]])
                ELSE ''
            END as content,
            toString(COALESCE(node.source_file, node.source, '')) as source_file,
            toString(COALESCE(node.source_path, '')) as source_path,
            COALESCE(toInteger(node.page), -1) as page,
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

    def find_node_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a node by exact or case-insensitive name match.
        """
        graph = self._get_graph()

        # Try exact match first on name property
        cypher_exact = """
        MATCH (n)
        WHERE n.name = $name
        RETURN n, labels(n) as labels
        LIMIT 1
        """
        results = graph.query(cypher_exact, params={"name": name})
        if results:
            node_data = dict(results[0]["n"])
            node_data["labels"] = results[0]["labels"]
            return node_data

        # Try case-insensitive match
        cypher_fuzzy = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($name)
        RETURN n, labels(n) as labels
        LIMIT 1
        """
        results = graph.query(cypher_fuzzy, params={"name": name})
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
            COALESCE(neighbor.name, neighbor.id) as name,
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
            [n IN nodes(path) | {{id: n.id, name: COALESCE(n.name, n.id), labels: labels(n)}}] as nodes,
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
            print(f"Path finding failed: {e}")
            return None

    def bfs_search(
        self, start_id: str, end_id: str, max_depth: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find a path between two nodes using Bidirectional BFS.

        Strategy:
        1. Try APOC-based true BFS with terminator nodes (optimal)
        2. Fallback to bidirectional frontier expansion via two Cypher queries
        3. Final fallback to simple variable-length path match
        """
        graph = self._get_graph()

        # --- Strategy 1: APOC BFS (true breadth-first, optimal) ---
        try:
            apoc_cypher = f"""
            MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
            CALL apoc.path.expandConfig(start, {{
                terminatorNodes: [end],
                maxLevel: {max_depth},
                bfs: true,
                uniqueness: 'NODE_GLOBAL'
            }}) YIELD path
            RETURN
                [n IN nodes(path) | {{id: n.id, name: COALESCE(n.name, n.id), labels: labels(n)}}] as nodes,
                [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] as relationships
            ORDER BY length(path)
            LIMIT 1
            """
            results = graph.query(
                apoc_cypher, params={"start_id": start_id, "end_id": end_id}
            )
            if results:
                return {
                    "nodes": results[0]["nodes"],
                    "relationships": results[0]["relationships"],
                }
        except Exception:
            pass  # APOC not available, continue to fallback

        # --- Strategy 2: Bidirectional frontier expansion ---
        # Expand from both ends simultaneously, find overlap
        try:
            bidir_cypher = f"""
            MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
            WITH start, end
            OPTIONAL MATCH path_fwd = (start)-[*1..{max_depth // 2 + 1}]-(mid)
            WITH start, end, collect(DISTINCT mid) as fwd_frontier
            OPTIONAL MATCH path_bwd = (end)-[*1..{max_depth // 2 + 1}]-(mid2)
            WHERE mid2 IN fwd_frontier
            WITH start, end, mid2 as meeting_point
            WHERE meeting_point IS NOT NULL
            LIMIT 1
            MATCH path1 = shortestPath((start)-[*..{max_depth}]-(meeting_point))
            MATCH path2 = shortestPath((meeting_point)-[*..{max_depth}]-(end))
            WITH nodes(path1) + tail(nodes(path2)) as all_nodes,
                 relationships(path1) + relationships(path2) as all_rels
            RETURN
                [n IN all_nodes | {{id: n.id, name: COALESCE(n.name, n.id), labels: labels(n)}}] as nodes,
                [r IN all_rels | {{type: type(r), properties: properties(r)}}] as relationships
            LIMIT 1
            """
            results = graph.query(
                bidir_cypher, params={"start_id": start_id, "end_id": end_id}
            )
            if results and results[0].get("nodes"):
                return {
                    "nodes": results[0]["nodes"],
                    "relationships": results[0]["relationships"],
                }
        except Exception:
            pass  # Continue to final fallback

        # --- Strategy 3: Simple variable-length path (original fallback) ---
        cypher = f"""
        MATCH path = (start {{id: $start_id}})-[*1..{max_depth}]-(end {{id: $end_id}})
        RETURN
            [n IN nodes(path) | {{id: n.id, name: COALESCE(n.name, n.id), labels: labels(n)}}] as nodes,
            [r IN relationships(path) | {{type: type(r), properties: properties(r)}}] as relationships
        ORDER BY length(path)
        LIMIT 1
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
            print(f"BFS search failed: {e}")
            return None

    def parallel_bfs_from_seeds(
        self, seed_ids: List[str], max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Launch parallel BFS expansions from multiple seed nodes.
        Merges discovered subgraphs, deduplicates, and identifies bridge nodes.

        Args:
            seed_ids: List of entity IDs to expand from
            max_depth: How far to expand from each seed

        Returns:
            {
                "nodes": deduplicated list of all discovered nodes,
                "relationships": deduplicated list of all discovered relationships,
                "bridge_nodes": nodes found in multiple seed expansions,
                "seed_subgraphs": {seed_id: subgraph} mapping
            }
        """
        if not seed_ids:
            return {
                "nodes": [],
                "relationships": [],
                "bridge_nodes": [],
                "seed_subgraphs": {},
            }

        # Deduplicate seed IDs
        seed_ids = list(set(seed_ids))

        def _expand_single_seed(seed_id: str) -> Dict[str, Any]:
            """Expand from a single seed node."""
            graph = self._get_graph()

            # Try APOC first for true BFS
            try:
                apoc_cypher = f"""
                MATCH (center {{id: $seed_id}})
                CALL apoc.path.subgraphAll(center, {{maxLevel: {max_depth}}})
                YIELD nodes, relationships
                RETURN
                    [n IN nodes | {{
                        id: n.id,
                        name: COALESCE(n.name, n.id),
                        labels: labels(n),
                        description: n.description
                    }}] as nodes,
                    [r IN relationships | {{
                        type: type(r),
                        start: startNode(r).id,
                        end: endNode(r).id
                    }}] as relationships
                """
                results = graph.query(apoc_cypher, params={"seed_id": seed_id})
                if results:
                    return {
                        "nodes": results[0]["nodes"],
                        "relationships": results[0]["relationships"],
                    }
            except Exception:
                pass  # APOC not available

            # Fallback: variable-length path expansion
            fallback_nodes_cypher = f"""
            MATCH (center {{id: $seed_id}})
            OPTIONAL MATCH (center)-[*0..{max_depth}]-(connected)
            WITH connected
            WHERE connected IS NOT NULL
            RETURN
                collect(DISTINCT {{
                    id: connected.id,
                    name: COALESCE(connected.name, connected.id),
                    labels: labels(connected),
                    description: connected.description
                }}) as nodes
            """
            fallback_relationships_cypher = f"""
            MATCH (center {{id: $seed_id}})
            OPTIONAL MATCH path = (center)-[r*1..{max_depth}]-()
            UNWIND r as rel
            RETURN
                collect(DISTINCT {{
                    type: type(rel),
                    start: startNode(rel).id,
                    end: endNode(rel).id
                }}) as relationships
            """
            try:
                node_results = graph.query(
                    fallback_nodes_cypher, params={"seed_id": seed_id}
                )
                rel_results = graph.query(
                    fallback_relationships_cypher, params={"seed_id": seed_id}
                )
                return {
                    "nodes": node_results[0]["nodes"] if node_results else [],
                    "relationships": rel_results[0]["relationships"]
                    if rel_results
                    else [],
                }
            except Exception as e:
                print(f"Seed expansion failed for {seed_id}: {e}")
                return {"nodes": [], "relationships": []}

        # Run expansions in parallel
        seed_subgraphs = {}
        max_workers = min(len(seed_ids), config.MAX_PARALLEL_BFS_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(_expand_single_seed, sid): sid for sid in seed_ids
            }
            for future in as_completed(future_to_seed):
                seed_id = future_to_seed[future]
                try:
                    seed_subgraphs[seed_id] = future.result()
                except Exception as e:
                    print(f"Parallel BFS failed for seed {seed_id}: {e}")
                    seed_subgraphs[seed_id] = {"nodes": [], "relationships": []}

        # Merge and deduplicate
        all_nodes: Dict[str, Dict] = {}
        all_rels: List[Dict] = []
        node_occurrence: Dict[str, int] = {}  # How many seeds found this node
        seen_rels: Set[str] = set()

        for seed_id, subgraph in seed_subgraphs.items():
            for node in subgraph.get("nodes", []):
                nid = node.get("id")
                if nid:
                    all_nodes[nid] = node
                    node_occurrence[nid] = node_occurrence.get(nid, 0) + 1

            for rel in subgraph.get("relationships", []):
                rel_key = f"{rel.get('start')}_{rel.get('type')}_{rel.get('end')}"
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    all_rels.append(rel)

        # Bridge nodes: found from 2+ different seed expansions
        bridge_nodes = [
            all_nodes[nid]
            for nid, count in node_occurrence.items()
            if count >= 2 and nid not in seed_ids
        ]

        return {
            "nodes": list(all_nodes.values()),
            "relationships": all_rels,
            "bridge_nodes": bridge_nodes,
            "seed_subgraphs": seed_subgraphs,
        }

    def graph_rerank(
        self,
        entities: List[Dict[str, Any]],
        query_entity_ids: List[str],
        vector_weight: float = None,
        graph_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank entity results by combining vector similarity with graph proximity.

        For each entity, computes:
            final_score = vector_weight * vector_score + graph_weight * graph_proximity_score

        Graph proximity is measured by shortest path distance to any query entity.
        """
        if not entities or not query_entity_ids:
            return entities

        vector_weight = vector_weight or config.RERANK_VECTOR_WEIGHT
        graph_weight = graph_weight or config.RERANK_GRAPH_WEIGHT

        graph = self._get_graph()
        entity_ids = [e.get("id") for e in entities if e.get("id")]
        query_id_set = {qid for qid in query_entity_ids if qid}
        distance_map = {eid: 0 for eid in entity_ids if eid in query_id_set}
        query_ids_for_db = [qid for qid in query_id_set]
        entity_ids_for_db = [eid for eid in entity_ids if eid not in query_id_set]

        # Batch query: get shortest path lengths from each entity to any query entity
        proximity_cypher = """
        UNWIND $entity_ids as eid
        UNWIND $query_ids as qid
        OPTIONAL MATCH path = shortestPath(
            (e {id: eid})-[*..5]-(q {id: qid})
        )
        WITH eid, MIN(CASE WHEN path IS NOT NULL THEN length(path) ELSE null END) as min_distance
        RETURN eid, min_distance
        """

        try:
            if entity_ids_for_db and query_ids_for_db:
                results = graph.query(
                    proximity_cypher,
                    params={
                        "entity_ids": entity_ids_for_db,
                        "query_ids": query_ids_for_db,
                    },
                )

                # Build distance map
                for row in results:
                    eid = row.get("eid")
                    dist = row.get("min_distance")
                    if eid is not None:
                        distance_map[eid] = dist

        except Exception as e:
            print(f"Graph reranking proximity query failed: {e}")
            # Fall back to pure vector ranking
            return entities

        # Compute blended scores
        max_distance = 6  # Normalize distances: anything >= 6 hops gets proximity 0
        reranked = []

        for entity in entities:
            eid = entity.get("id")
            vector_score = entity.get("score", 0.0)

            dist = distance_map.get(eid)
            if dist is not None:
                # Proximity score: 1.0 for distance 0 (same node), decreasing
                graph_proximity = max(0.0, 1.0 - (dist / max_distance))
            else:
                graph_proximity = 0.0  # No path found

            blended_score = (vector_weight * vector_score) + (
                graph_weight * graph_proximity
            )

            reranked_entity = dict(entity)
            reranked_entity["original_score"] = vector_score
            reranked_entity["graph_proximity"] = graph_proximity
            reranked_entity["score"] = round(blended_score, 4)
            reranked.append(reranked_entity)

        # Sort by blended score descending
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

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
                name: COALESCE(n.name, n.id), 
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
        MATCH (center {{id: $center_id}})
        OPTIONAL MATCH (center)-[*0..{max_depth}]-(connected)
        WITH center, collect(DISTINCT {{
            id: connected.id,
            name: COALESCE(connected.name, connected.id),
            labels: labels(connected),
            description: connected.description
        }}) as nodes
        CALL {{
            WITH center
            OPTIONAL MATCH path = (center)-[r*1..{max_depth}]-()
            UNWIND r as rel
            RETURN collect(DISTINCT {{
                type: type(rel),
                start: startNode(rel).id,
                end: endNode(rel).id,
                properties: properties(rel)
            }}) as relationships
        }}
        RETURN nodes, relationships
        """

        results = graph.query(fallback_cypher, params={"center_id": center_id})
        if results:
            return {
                "nodes": results[0].get("nodes", []),
                "relationships": results[0].get("relationships", []),
            }
        return {"nodes": [], "relationships": []}

    # =========================================================================
    # HYBRID QUERY
    # =========================================================================

    def _decompose_question(self, question: str) -> List[str]:
        """
        Use an LLM planner to split compound prompts into independent sub-questions.
        Returns at most 15 questions for downstream retrieval.
        """
        max_sub_questions = 15
        normalized = " ".join((question or "").split())
        if not normalized:
            return []

        planner_prompt = f"""You are a query planning assistant for a graph RAG system.

Task:
Split the user's message into atomic sub-questions for retrieval.

Rules:
1) Output STRICT JSON only with this shape:
   {{"sub_questions": ["..."]}}
2) Keep dependent clauses together when they belong to one fact request.
   Example: "How much did Google invest in SpaceX and for how much ownership?"
   should stay as ONE sub-question.
3) Preserve original meaning and wording as much as possible.
4) Return between 1 and {max_sub_questions} sub-questions.
5) Remove duplicates.
6) No markdown, no explanations.

User question:
{normalized}
"""

        try:
            llm = self._get_agent_llm("subquery_planner")
            response = llm.invoke(planner_prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    for item in content
                )

            text = str(content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()

            parsed = json.loads(text)
            if isinstance(parsed, dict):
                candidates = parsed.get("sub_questions", [])
            elif isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = []

            deduped: List[str] = []
            seen = set()
            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                cleaned = candidate.strip(" ,.;")
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(cleaned)

            return deduped[:max_sub_questions] if deduped else [normalized]
        except Exception:
            # Safety fallback when planner output is malformed/unavailable.
            return [normalized]

    def _plan_document_retrieval(self, question: str) -> Dict[str, Any]:
        """
        Use an LLM planner to generate retrieval strategy for document chunks.
        Returns semantic query variants and lexical hints without hard-coded rules.
        """
        normalized = " ".join((question or "").split()).strip()
        if not normalized:
            return {"semantic_queries": [], "lexical_keywords": [], "numeric_focus": False}

        planner_prompt = f"""You are a retrieval planner for a graph RAG system.

Task:
Produce a retrieval plan for the user's question.

Output STRICT JSON only with this shape:
{{
  "semantic_queries": ["..."],
  "lexical_keywords": ["..."],
  "min_lexical_hits": 2,
  "numeric_focus": false,
  "rerank_terms": ["..."]
}}

Rules:
1) Keep semantic_queries concise and high-signal (1-4 entries).
2) lexical_keywords should be short literal tokens/phrases useful for exact text matching.
3) Set numeric_focus=true if the question requests exact numbers, amounts, percentages, counts, dates, or rankings.
4) min_lexical_hits must be an integer between 1 and 4.
5) rerank_terms should contain the most discriminative evidence terms.
6) No markdown. No explanation text.

User question:
{normalized}
"""

        fallback_plan = {
            "semantic_queries": [normalized],
            "lexical_keywords": [],
            "min_lexical_hits": 2,
            "numeric_focus": False,
            "rerank_terms": [],
        }

        try:
            llm = self._get_agent_llm("retrieval_planner")
            response = llm.invoke(planner_prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    for item in content
                )

            text = str(content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()

            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return fallback_plan

            semantic_queries = parsed.get("semantic_queries", [])
            lexical_keywords = parsed.get("lexical_keywords", [])
            rerank_terms = parsed.get("rerank_terms", [])
            min_lexical_hits = parsed.get("min_lexical_hits", 2)
            numeric_focus = bool(parsed.get("numeric_focus", False))

            if not isinstance(semantic_queries, list):
                semantic_queries = []
            if not isinstance(lexical_keywords, list):
                lexical_keywords = []
            if not isinstance(rerank_terms, list):
                rerank_terms = []

            def _clean_strings(items: List[Any], max_items: int) -> List[str]:
                cleaned: List[str] = []
                seen = set()
                for item in items:
                    if not isinstance(item, str):
                        continue
                    value = item.strip()
                    if not value:
                        continue
                    key = value.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(value)
                    if len(cleaned) >= max_items:
                        break
                return cleaned

            semantic_queries = _clean_strings(semantic_queries, 4)
            lexical_keywords = _clean_strings(lexical_keywords, 12)
            rerank_terms = _clean_strings(rerank_terms, 12)

            if not semantic_queries:
                semantic_queries = [normalized]

            try:
                min_lexical_hits = int(min_lexical_hits)
            except Exception:
                min_lexical_hits = 2
            min_lexical_hits = min(4, max(1, min_lexical_hits))

            return {
                "semantic_queries": semantic_queries,
                "lexical_keywords": lexical_keywords,
                "min_lexical_hits": min_lexical_hits,
                "numeric_focus": numeric_focus,
                "rerank_terms": rerank_terms,
            }
        except Exception:
            return fallback_plan

    def _curate_document_evidence(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 12,
    ) -> Dict[str, Any]:
        """
        Evidence curator agent:
        Select and order the most relevant chunks and extract concise evidence snippets.
        """
        if not chunks:
            return {"chunks": [], "evidence_snippets": []}

        candidates = []
        for chunk in chunks[:30]:
            content = str(chunk.get("content", "") or "")
            preview = content[:600] + ("..." if len(content) > 600 else "")
            candidates.append(
                {
                    "id": str(chunk.get("id", "")),
                    "source": self._format_chunk_source(chunk),
                    "score": float(chunk.get("score", 0) or 0),
                    "preview": preview,
                }
            )

        planner_prompt = f"""You are an evidence curator agent for a graph RAG pipeline.

Question:
{question}

Candidate chunks (JSON):
{json.dumps(candidates, ensure_ascii=True)}

Task:
1) Select the chunk ids that contain the strongest direct evidence for answering the question.
2) Order selected_chunk_ids from strongest to weakest.
3) Provide short evidence snippets tied to selected chunks.

Output STRICT JSON:
{{
  "selected_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "evidence_snippets": [
    {{"chunk_id": "chunk_id_1", "text": "short evidence statement"}}
  ]
}}

Rules:
- Return 1 to {max_chunks} selected_chunk_ids.
- Use only ids from the candidate list.
- Snippets must be concise (<= 220 chars each) and evidence-grounded.
- No markdown, no explanations outside JSON.
"""

        fallback_chunks = sorted(
            [dict(item) for item in chunks],
            key=lambda item: float(item.get("score", 0) or 0),
            reverse=True,
        )[:max_chunks]

        try:
            llm = self._get_agent_llm("evidence_curator")
            response = llm.invoke(planner_prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else str(item)
                    )
                    for item in content
                )

            text = str(content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json", "", 1).strip()

            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return {"chunks": fallback_chunks, "evidence_snippets": []}

            selected_ids_raw = parsed.get("selected_chunk_ids", [])
            snippets_raw = parsed.get("evidence_snippets", [])

            chunk_map = {str(chunk.get("id", "")): dict(chunk) for chunk in chunks}
            selected_ids: List[str] = []
            seen = set()
            for item in selected_ids_raw if isinstance(selected_ids_raw, list) else []:
                if not isinstance(item, str):
                    continue
                key = item.strip()
                if not key or key in seen or key not in chunk_map:
                    continue
                seen.add(key)
                selected_ids.append(key)
                if len(selected_ids) >= max_chunks:
                    break

            curated_chunks = [chunk_map[item] for item in selected_ids]
            if not curated_chunks:
                curated_chunks = fallback_chunks
            elif len(curated_chunks) < max_chunks:
                for chunk in fallback_chunks:
                    chunk_id = str(chunk.get("id", ""))
                    if chunk_id and chunk_id not in selected_ids:
                        curated_chunks.append(chunk)
                    if len(curated_chunks) >= max_chunks:
                        break

            evidence_snippets: List[str] = []
            if isinstance(snippets_raw, list):
                for snippet in snippets_raw:
                    if not isinstance(snippet, dict):
                        continue
                    chunk_id = str(snippet.get("chunk_id", "")).strip()
                    text_value = str(snippet.get("text", "")).strip()
                    if not chunk_id or not text_value:
                        continue
                    if chunk_id not in chunk_map:
                        continue
                    source = self._format_chunk_source(chunk_map[chunk_id])
                    short_text = text_value[:220] + ("..." if len(text_value) > 220 else "")
                    bullet = f"- {short_text} (source: {source})"
                    if bullet not in evidence_snippets:
                        evidence_snippets.append(bullet)
                    if len(evidence_snippets) >= max_chunks:
                        break

            return {"chunks": curated_chunks[:max_chunks], "evidence_snippets": evidence_snippets}
        except Exception:
            return {"chunks": fallback_chunks, "evidence_snippets": []}

    def _keyword_search_documents(
        self, keywords: List[str], top_k: int = 8, min_hits: int = 2
    ) -> List[Dict[str, Any]]:
        """Fallback lexical search over document chunks for high-precision fact retrieval."""
        cleaned_keywords = []
        seen = set()
        for keyword in keywords:
            kw = keyword.strip().lower()
            if len(kw) < 3 or kw in seen:
                continue
            seen.add(kw)
            cleaned_keywords.append(kw)

        if not cleaned_keywords:
            return []

        graph = self._get_graph()
        cypher = """
        MATCH (d:Document)
        WITH d, [k IN ['text', 'page_content', 'content'] WHERE k IN keys(d)] as content_keys
        WITH d, CASE
            WHEN size(content_keys) > 0 THEN toString(d[content_keys[0]])
            ELSE ''
        END as raw_content
        WITH d, raw_content, toLower(raw_content) as lowered_content
        WITH d, raw_content, lowered_content,
             reduce(
                score = 0,
                kw IN $keywords |
                score + CASE WHEN lowered_content CONTAINS kw THEN 1 ELSE 0 END
             ) as hit_count
        WHERE hit_count >= $min_hits
        RETURN
            d.id as id,
            raw_content as content,
            toString(COALESCE(d.source_file, d.source, '')) as source_file,
            toString(COALESCE(d.source_path, '')) as source_path,
            COALESCE(toInteger(d.page), -1) as page,
            toFloat(hit_count) / toFloat(size($keywords)) as score
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            results = graph.query(
                cypher,
                params={
                    "keywords": cleaned_keywords,
                    "min_hits": max(1, min_hits),
                    "top_k": max(1, top_k),
                },
            )
            return results or []
        except Exception as e:
            print(f"Keyword document search failed: {e}")
            return []

    def _merge_evidence_snippets(
        self, grouped_snippets: List[List[str]], max_snippets: int = 12
    ) -> List[str]:
        """Merge and deduplicate evidence snippets from multiple sub-query contexts."""
        merged: List[str] = []
        seen = set()

        for snippets in grouped_snippets:
            for snippet in snippets:
                if not isinstance(snippet, str):
                    continue
                clean = snippet.strip()
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                merged.append(clean)
                if len(merged) >= max_snippets:
                    return merged

        return merged

    def _retrieve_hybrid_context(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """Run retrieval/expansion/reranking for one question segment."""
        entities = self.vector_search(
            question, top_k=top_k, score_threshold=config.VECTOR_SEARCH_SCORE_THRESHOLD
        )
        retrieval_plan = self._plan_document_retrieval(question)
        doc_query_variants = retrieval_plan.get("semantic_queries", []) or [question]
        doc_chunk_groups = [
            self.vector_search_documents(query_variant, top_k=7)
            for query_variant in doc_query_variants
        ]
        lexical_keywords = retrieval_plan.get("lexical_keywords", [])
        if lexical_keywords:
            lexical_hits = self._keyword_search_documents(
                lexical_keywords,
                top_k=8,
                min_hits=retrieval_plan.get("min_lexical_hits", 2),
            )
            if lexical_hits:
                doc_chunk_groups.append(lexical_hits)

        doc_chunks = self._merge_doc_chunks(doc_chunk_groups, max_chunks=20)
        curation = self._curate_document_evidence(
            question, doc_chunks, max_chunks=12
        )
        doc_chunks = curation.get("chunks", []) or []
        evidence_snippets = curation.get("evidence_snippets", []) or []

        seed_ids = [e["id"] for e in entities[:5] if e.get("id")]
        expanded_subgraph = {"nodes": [], "relationships": [], "bridge_nodes": []}

        if seed_ids:
            try:
                expanded_subgraph = self.parallel_bfs_from_seeds(
                    seed_ids, max_depth=config.MULTI_HOP_CONTEXT_DEPTH
                )
            except Exception as e:
                print(f"Parallel BFS expansion failed, falling back to neighbors: {e}")
                for entity in entities[:3]:
                    neighbors = self.get_neighbors(
                        entity["id"], depth=config.HYBRID_SEARCH_DEPTH
                    )
                    expanded_subgraph["nodes"].extend(neighbors)

        if entities and seed_ids:
            try:
                entities = self.graph_rerank(
                    entities,
                    query_entity_ids=seed_ids[:3],
                    vector_weight=config.RERANK_VECTOR_WEIGHT,
                    graph_weight=config.RERANK_GRAPH_WEIGHT,
                )
            except Exception as e:
                print(f"Graph reranking failed, using vector-only ranking: {e}")

        return {
            "question": question,
            "entities": entities,
            "doc_chunks": doc_chunks,
            "evidence_snippets": evidence_snippets,
            "expanded_subgraph": expanded_subgraph,
        }

    def _merge_entities(
        self, grouped_entities: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate entities while preserving the strongest match."""
        merged: Dict[str, Dict[str, Any]] = {}

        for entities in grouped_entities:
            for entity in entities:
                entity_id = entity.get("id") or entity.get("name")
                if not entity_id:
                    continue

                key = str(entity_id)
                candidate = dict(entity)
                existing = merged.get(key)

                if existing is None:
                    merged[key] = candidate
                    continue

                if candidate.get("score", 0) > existing.get("score", 0):
                    merged[key] = candidate
                    continue

                if not existing.get("description") and candidate.get("description"):
                    existing["description"] = candidate["description"]
                if (
                    existing.get("graph_proximity") is None
                    and candidate.get("graph_proximity") is not None
                ):
                    existing["graph_proximity"] = candidate["graph_proximity"]
                if not existing.get("labels") and candidate.get("labels"):
                    existing["labels"] = candidate["labels"]

        return sorted(merged.values(), key=lambda item: item.get("score", 0), reverse=True)

    def _merge_doc_chunks(
        self, grouped_chunks: List[List[Dict[str, Any]]], max_chunks: int
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate document chunks by chunk id/content."""
        merged: Dict[str, Dict[str, Any]] = {}

        def _chunk_rank_value(chunk: Dict[str, Any]) -> float:
            if chunk.get("_hybrid_score") is not None:
                return float(chunk.get("_hybrid_score", 0) or 0)
            return float(chunk.get("score", 0) or 0)

        for chunks in grouped_chunks:
            for chunk in chunks:
                chunk_id = chunk.get("id")
                if not chunk_id:
                    content = str(chunk.get("content", ""))
                    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
                    chunk_id = f"content:{content_hash}"
                key = str(chunk_id)

                existing = merged.get(key)
                if (
                    existing is None
                    or _chunk_rank_value(chunk) > _chunk_rank_value(existing)
                ):
                    merged[key] = dict(chunk)

        deduped = sorted(merged.values(), key=_chunk_rank_value, reverse=True)
        return deduped[:max_chunks]

    def _merge_expanded_subgraphs(
        self, subgraphs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge nodes/relationships/bridge nodes from multiple expansions."""
        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        relationships_by_key: Dict[str, Dict[str, Any]] = {}
        bridge_by_id: Dict[str, Dict[str, Any]] = {}

        for subgraph in subgraphs:
            for node in subgraph.get("nodes", []) or []:
                node_id = node.get("id")
                if not node_id:
                    continue
                nodes_by_id.setdefault(str(node_id), dict(node))

            for bridge in subgraph.get("bridge_nodes", []) or []:
                bridge_id = bridge.get("id")
                if not bridge_id:
                    continue
                bridge_by_id.setdefault(str(bridge_id), dict(bridge))

            for rel in subgraph.get("relationships", []) or []:
                start = rel.get("start", "")
                rel_type = rel.get("type", "RELATED_TO")
                end = rel.get("end", "")
                key = f"{start}|{rel_type}|{end}"
                if key not in relationships_by_key:
                    relationships_by_key[key] = dict(rel)

        return {
            "nodes": list(nodes_by_id.values()),
            "relationships": list(relationships_by_key.values()),
            "bridge_nodes": list(bridge_by_id.values()),
        }

    def _format_chunk_source(self, chunk: Dict[str, Any]) -> str:
        """Create a human-readable source label for a document chunk."""
        source_name = chunk.get("source_file") or chunk.get("source_path") or chunk.get("id")
        page = chunk.get("page", -1)
        if isinstance(page, int) and page >= 0:
            return f"{source_name} p.{page + 1}"
        return str(source_name)

    def _synthesize_hybrid_answer(self, question: str, context_str: str) -> str:
        """Answer synthesizer agent for final response generation."""
        llm = self._get_agent_llm("answer_synthesizer")
        prompt = f"""Based on the following knowledge from a knowledge graph and source documents, answer the user's question.

The information includes:
- Source documents (raw text from original files)
- Key entities found via semantic search, ranked by combined vector similarity and graph proximity
- Bridge entities that connect multiple topics in the knowledge graph
- Expanded graph context showing related entities discovered through graph traversal
- Relationship chains showing how entities are connected

{context_str}

Question: {question}

Instructions:
- If the question contains multiple parts, answer each part explicitly in a numbered list.
- Prefer exact values from source documents when available.
- Include source attribution like "(source: SpaceX.pdf p.5)" for key facts when possible.
- If a part of the question is not grounded in retrieved evidence, state that clearly.

Answer:"""
        response = llm.invoke(prompt)
        return response.content

    def query_hybrid(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        SOTA Hybrid query combining vector search, parallel BFS expansion,
        graph-aware reranking, and LLM synthesis.

        Pipeline:
        1. RETRIEVE  - Vector search entities + document chunks
        2. EXPAND    - Parallel BFS from top-K entity seeds
        3. RERANK    - Blend vector similarity with graph proximity
        4. SYNTHESIZE - Feed reranked, expanded context to LLM
        """
        sub_questions = self._decompose_question(question)
        contexts = [
            self._retrieve_hybrid_context(sub_question, top_k=top_k)
            for sub_question in sub_questions
        ]
        contexts_with_hits = [
            ctx for ctx in contexts if ctx.get("entities") or ctx.get("doc_chunks")
        ]

        if not contexts_with_hits:
            return {
                "answer": self.query(question),
                "entities_found": [],
                "graph_context": [],
                "bridge_nodes": [],
                "sub_questions": sub_questions,
                "sub_question_coverage": 0.0,
                "confidence": 0.3,
                "method": "cypher_fallback",
            }

        entities = self._merge_entities(
            [ctx.get("entities", []) for ctx in contexts_with_hits]
        )
        doc_chunk_budget = (
            min(18, max(8, len(sub_questions) * 4)) if len(sub_questions) > 1 else 6
        )
        doc_chunks = self._merge_doc_chunks(
            [ctx.get("doc_chunks", []) for ctx in contexts_with_hits],
            max_chunks=doc_chunk_budget,
        )
        evidence_snippets = self._merge_evidence_snippets(
            [ctx.get("evidence_snippets", []) for ctx in contexts_with_hits],
            max_snippets=min(15, max(6, len(sub_questions) * 3)),
        )
        expanded_subgraph = self._merge_expanded_subgraphs(
            [ctx.get("expanded_subgraph", {}) for ctx in contexts_with_hits]
        )
        sub_question_coverage = len(contexts_with_hits) / max(len(sub_questions), 1)

        # === Stage 4: SYNTHESIZE ===
        context_parts = []

        if len(sub_questions) > 1:
            context_parts.append("## Decomposed Sub-Questions:")
            for index, sub_question in enumerate(sub_questions, start=1):
                context_parts.append(f"{index}. {sub_question}")

        # Document chunks (highest fidelity source)
        if doc_chunks:
            context_parts.append("## Relevant Source Documents:")
            doc_limit = min(len(doc_chunks), max(4, len(sub_questions) * 3))
            content_limit = 500 if len(sub_questions) >= 4 else 800
            for chunk in doc_chunks[:doc_limit]:
                content = chunk.get("content", "")
                if content:
                    content = (
                        content[:content_limit] + "..."
                        if len(content) > content_limit
                        else content
                    )
                    source = self._format_chunk_source(chunk)
                    context_parts.append(f"Source: {source}\n```\n{content}\n```")
            if evidence_snippets:
                context_parts.append("\n## Evidence Snippets:")
                context_parts.extend(evidence_snippets)

        # Reranked entity context
        if entities:
            context_parts.append(
                "\n## Key Entities Found (ranked by relevance + graph proximity):"
            )
            for ent in entities[:5]:
                labels = ", ".join(
                    [label for label in ent.get("labels", []) if label != "__Entity__"]
                )
                desc = ent.get("description") or "No description available"
                name = ent.get("name", ent.get("id", "Unknown"))
                score_info = f"score={ent.get('score', 0):.2f}"
                if ent.get("graph_proximity") is not None:
                    score_info += f", graph_proximity={ent['graph_proximity']:.2f}"
                context_parts.append(f"- **{name}** ({labels}, {score_info}): {desc}")

        # Bridge nodes (entities connecting multiple seed entities)
        bridge_nodes = expanded_subgraph.get("bridge_nodes", [])
        if bridge_nodes:
            context_parts.append("\n## Bridge Entities (connecting multiple topics):")
            for bn in bridge_nodes[:5]:
                name = bn.get("name", bn.get("id", "Unknown"))
                labels = ", ".join(
                    [label for label in bn.get("labels", []) if label != "__Entity__"]
                )
                desc = bn.get("description") or ""
                context_parts.append(f"- **{name}** ({labels}): {desc}")

        # Expanded graph context (discovered via BFS)
        expanded_nodes = expanded_subgraph.get("nodes", [])
        if expanded_nodes:
            context_parts.append("\n## Expanded Graph Context:")
            seen = set(e.get("id") for e in entities[:5])  # Skip already-shown entities
            count = 0
            for node in expanded_nodes:
                nid = node.get("id")
                if nid and nid not in seen and count < 10:
                    seen.add(nid)
                    name = node.get("name", nid)
                    labels = ", ".join(
                        [
                            label
                            for label in node.get("labels", [])
                            if label != "__Entity__"
                        ]
                    )
                    desc = node.get("description") or ""
                    if desc:
                        context_parts.append(f"- {name} ({labels}): {desc}")
                    else:
                        context_parts.append(f"- {name} ({labels})")
                    count += 1

        # Relationship chains
        expanded_rels = expanded_subgraph.get("relationships", [])
        if expanded_rels:
            context_parts.append("\n## Relationship Chains:")
            for rel in expanded_rels[:15]:
                start = rel.get("start", "?")
                end = rel.get("end", "?")
                rel_type = rel.get("type", "RELATED_TO")
                context_parts.append(f"- {start} --[{rel_type}]--> {end}")

        context_str = "\n".join(context_parts)

        llm_failed = False
        try:
            answer = self._synthesize_hybrid_answer(question, context_str)
        except Exception as e:
            llm_failed = True
            answer = f"Error generating answer: {e}"

        # Confidence: blend vector relevance, document relevance, and multi-part coverage.
        top_entity_score = entities[0].get("score", 0) if entities else 0
        top_doc_score = doc_chunks[0].get("score", 0) if doc_chunks else 0
        confidence = (
            0.6 * top_entity_score + 0.3 * top_doc_score + 0.1 * sub_question_coverage
        )

        if not entities and doc_chunks:
            confidence = min(max(top_doc_score, 0.4) * sub_question_coverage, 0.85)

        if bridge_nodes:
            confidence = min(confidence * 1.05, 1.0)

        if llm_failed:
            confidence = min(confidence, 0.2)

        return {
            "answer": answer,
            "entities_found": entities,
            "graph_context": expanded_nodes[:10],
            "bridge_nodes": bridge_nodes,
            "doc_chunks_used": len(doc_chunks),
            "sub_questions": sub_questions,
            "sub_question_coverage": round(sub_question_coverage, 3),
            "confidence": round(confidence, 3),
            "method": "hybrid_sota",
            "status": "error" if llm_failed else "success",
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

