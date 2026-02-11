"""
Neo4j Repository - Comprehensive graph database operations with vector search and traversal.
Enhanced with SOTA GraphRAG: bidirectional BFS, parallel multi-source BFS, graph-aware reranking.
"""

import hashlib
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
        # Ensure connection is alive before processing batch
        self.verify_connectivity()
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
                nodes_to_embed = {}  # id -> {name, description}
                for g_doc in batch:
                    for node in g_doc.nodes:
                        if node.id not in nodes_to_embed:
                            # Extract description from properties if available
                            name = str(node.id)
                            description = ""
                            if hasattr(node, "properties") and node.properties:
                                name = node.properties.get("name") or name
                                description = node.properties.get("description", "")
                            nodes_to_embed[node.id] = {
                                "name": name,
                                "description": description,
                            }

                if nodes_to_embed:
                    # Create embedding text: "name: description" for better semantic matching
                    node_ids = list(nodes_to_embed.keys())
                    texts_to_embed = []
                    for nid in node_ids:
                        name = nodes_to_embed[nid]["name"]
                        desc = nodes_to_embed[nid]["description"]
                        if desc:
                            texts_to_embed.append(f"{name}: {desc}")
                        else:
                            texts_to_embed.append(name)

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
                        metadata = g_doc.source.metadata or {}
                        source_name = metadata.get("source") or metadata.get(
                            "source_file", "unknown"
                        )
                        page = metadata.get("page", 0)
                        chunk_id = metadata.get("chunk_id") or metadata.get("id")
                        doc_content = g_doc.source.page_content
                        if not chunk_id and doc_content:
                            content_hash = hashlib.md5(
                                doc_content.encode("utf-8")
                            ).hexdigest()[:12]
                            chunk_id = f"{source_name}#p{page}:{content_hash}"
                        if chunk_id and doc_content:
                            docs_to_embed.append(
                                {"id": chunk_id, "content": doc_content}
                            )

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
        # === Stage 1: RETRIEVE ===
        entities = self.vector_search(
            question, top_k=top_k, score_threshold=config.VECTOR_SEARCH_SCORE_THRESHOLD
        )
        doc_chunks = self.vector_search_documents(question, top_k=5)

        if not entities and not doc_chunks:
            return {
                "answer": self.query(question),
                "entities_found": [],
                "graph_context": [],
                "bridge_nodes": [],
                "confidence": 0.3,
                "method": "cypher_fallback",
            }

        # === Stage 2: EXPAND via parallel BFS ===
        seed_ids = [e["id"] for e in entities[:5] if e.get("id")]  # Top 5 as seeds
        expanded_subgraph = {"nodes": [], "relationships": [], "bridge_nodes": []}

        if seed_ids:
            try:
                expanded_subgraph = self.parallel_bfs_from_seeds(
                    seed_ids, max_depth=config.MULTI_HOP_CONTEXT_DEPTH
                )
            except Exception as e:
                print(f"Parallel BFS expansion failed, falling back to neighbors: {e}")
                # Fallback to simple neighbor expansion
                for entity in entities[:3]:
                    neighbors = self.get_neighbors(
                        entity["id"], depth=config.HYBRID_SEARCH_DEPTH
                    )
                    expanded_subgraph["nodes"].extend(neighbors)

        # === Stage 3: RERANK with graph proximity ===
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

        # === Stage 4: SYNTHESIZE ===
        context_parts = []

        # Document chunks (highest fidelity source)
        if doc_chunks:
            context_parts.append("## Relevant Source Documents:")
            for chunk in doc_chunks[:3]:
                content = chunk.get("content", "")
                if content:
                    content = content[:800] + "..." if len(content) > 800 else content
                    context_parts.append(f"```\n{content}\n```")

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

        # LLM synthesis
        llm = ChatOpenAI(temperature=0, model_name=config.LLM_MODEL)

        prompt = f"""Based on the following knowledge from a knowledge graph and source documents, answer the question.

The information includes:
- Source documents (raw text from original files)
- Key entities found via semantic search, ranked by combined vector similarity and graph proximity
- Bridge entities that connect multiple topics in the knowledge graph
- Expanded graph context showing related entities discovered through graph traversal
- Relationship chains showing how entities are connected

{context_str}

Question: {question}

Provide a clear, comprehensive answer. Prioritize information from source documents for details and use the knowledge graph structure for understanding relationships and connections. If bridge entities are relevant, explain how they connect the topics.

Answer:"""

        llm_failed = False
        try:
            response = llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            llm_failed = True
            answer = f"Error generating answer: {e}"

        # Confidence: blend of vector score, graph proximity, and doc coverage
        if entities:
            top_score = entities[0].get("score", 0)
            has_bridges = len(bridge_nodes) > 0
            has_docs = len(doc_chunks) > 0
            confidence = min(
                top_score * (1.1 if has_bridges else 1.0) * (1.1 if has_docs else 1.0),
                1.0,
            )
        elif doc_chunks:
            confidence = min(doc_chunks[0].get("score", 0.5), 0.8)
        else:
            confidence = 0.3

        if llm_failed:
            confidence = min(confidence, 0.2)

        return {
            "answer": answer,
            "entities_found": entities,
            "graph_context": expanded_nodes[:10],
            "bridge_nodes": bridge_nodes,
            "doc_chunks_used": len(doc_chunks),
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
