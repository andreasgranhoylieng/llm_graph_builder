"""
TraversalService - Advanced graph traversal and context extraction for RAG.
"""

import re
from typing import List, Dict, Any, Optional
from src.services.interfaces import INeo4jRepository, ITraversalService
from src import config


class TraversalService(ITraversalService):
    """
    Service for advanced graph traversal operations.
    Provides methods for relationship discovery, context extraction, and similarity.
    """

    def __init__(self, neo4j_repo: INeo4jRepository):
        self.neo4j_repo = neo4j_repo

    def find_connections(
        self, entity_a: str, entity_b: str, max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        Find how two entities are connected in the knowledge graph.

        Args:
            entity_a: ID or name of first entity
            entity_b: ID or name of second entity
            max_depth: Maximum path length to search

        Returns:
            {
                "connected": bool,
                "path": List of nodes and relationships,
                "description": Human-readable description
            }
        """
        # Try to find direct path
        path = self.neo4j_repo.find_path(entity_a, entity_b, max_depth)

        if not path:
            # Fallback to BFS if shortest path fails (e.g. strict conditions or directionality)
            print(f"Shortest path failed, trying BFS for {entity_a} -> {entity_b}...")
            if hasattr(self.neo4j_repo, "bfs_search"):
                path = self.neo4j_repo.bfs_search(entity_a, entity_b, max_depth)

        if path:
            # Build human-readable description, filtering out Document nodes
            description_parts = []
            nodes = path.get("nodes", [])
            relationships = path.get("relationships", [])

            # Filter out Document nodes for cleaner display
            visible_nodes = []
            for node in nodes:
                labels = node.get("labels", [])
                # Skip Document nodes (they have hash IDs)
                if "Document" in labels:
                    continue
                visible_nodes.append(node)

            for i, node in enumerate(visible_nodes):
                # Get clean name, falling back to id
                name = node.get("name") or node.get("id", "Unknown")
                # Clean up hash-like names
                if len(name) == 32 and all(
                    c in "0123456789abcdef" for c in name.lower()
                ):
                    continue  # Skip hash IDs
                description_parts.append(name)
                if i < len(visible_nodes) - 1:
                    # Try to find the relationship between these nodes
                    if i < len(relationships):
                        rel_type = relationships[i].get("type", "RELATED_TO")
                        description_parts.append(f" --[{rel_type}]--> ")
                    else:
                        description_parts.append(" --[CONNECTED_TO]--> ")

            return {
                "connected": True,
                "path": path,
                "hops": len(visible_nodes) - 1 if visible_nodes else 0,
                "description": "".join(description_parts),
            }

        return {
            "connected": False,
            "path": None,
            "hops": -1,
            "description": f"No connection found between '{entity_a}' and '{entity_b}' within {max_depth} hops.",
        }

    def get_entity_context(
        self, entity_id: str, context_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get comprehensive context around an entity for RAG applications.

        This method returns:
        - The entity itself with all properties
        - Its immediate and extended neighbors
        - Relationship information
        - Connected documents/sources

        Args:
            entity_id: The ID of the entity to get context for
            context_depth: How many hops to include in context

        Returns:
            {
                "entity": dict,
                "neighbors": List of related entities,
                "relationships": List of relationship types,
                "sources": List of source documents,
                "context_summary": Human-readable summary
            }
        """
        # Get the main entity
        entity = self.neo4j_repo.get_node_by_id(entity_id)

        if not entity:
            return {
                "entity": None,
                "neighbors": [],
                "relationships": [],
                "sources": [],
                "context_summary": f"Entity '{entity_id}' not found.",
            }

        # Get subgraph
        subgraph = self.neo4j_repo.get_subgraph(entity_id, max_depth=context_depth)
        neighbors = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        # Extract unique relationship types
        rel_types = list(set(r.get("type") for r in relationships if r.get("type")))

        # Find connected documents
        sources = []
        for node in neighbors:
            if "Document" in node.get("labels", []):
                sources.append({"id": node.get("id"), "name": node.get("name")})

        # Build context summary
        summary_parts = [
            f"**{entity.get('name', entity_id)}**",
        ]

        if entity.get("description"):
            summary_parts.append(f": {entity['description']}")

        labels = entity.get("labels", [])
        if labels:
            summary_parts.append(f" (Type: {', '.join(labels)})")

        summary_parts.append(
            f"\n\nConnected to {len(neighbors)} entities via {len(rel_types)} relationship types."
        )

        if rel_types:
            summary_parts.append(f"\nRelationships: {', '.join(rel_types)}")

        return {
            "entity": entity,
            "neighbors": neighbors,
            "relationships": relationships,
            "relationship_types": rel_types,
            "sources": sources,
            "context_summary": "".join(summary_parts),
        }

    def find_similar_entities(
        self, entity_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to the given one using vector similarity.

        Args:
            entity_id: The reference entity ID
            top_k: Number of similar entities to return

        Returns:
            List of similar entities with similarity scores
        """
        # Get the entity to find its embedding text
        entity = self.neo4j_repo.get_node_by_id(entity_id)

        if not entity:
            return []

        # Use the entity's name and description for similarity search
        search_text = entity.get("name", "")
        if entity.get("description"):
            search_text += f": {entity['description']}"

        # Perform vector search
        similar = self.neo4j_repo.vector_search(
            search_text,
            top_k=top_k + 1,  # +1 because the entity itself might be returned
            score_threshold=0.5,
        )

        # Filter out the original entity
        filtered = [ent for ent in similar if ent.get("id") != entity_id][:top_k]

        return filtered

    def get_model_knowledge(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive knowledge about an AI model.

        Specialized method for the Generative AI domain that retrieves:
        - Model details
        - Developer/company
        - Architecture
        - Training data
        - Benchmarks
        - Related models
        """
        # Find the model (fuzzy search)
        models = self.neo4j_repo.vector_search(
            model_name, top_k=1, score_threshold=0.6, node_labels=["AIModel"]
        )

        if not models:
            # Relaxed fallback to handle legacy label drift (e.g. Aimodel)
            models = self.neo4j_repo.vector_search(
                model_name, top_k=3, score_threshold=0.75
            )
            models = [
                item
                for item in models
                if self._has_label(item, "AIModel")
                and self._token_overlap(model_name, item.get("name", "")) >= 0.4
            ][:1]

        if not models:
            return {"found": False, "message": f"Model '{model_name}' not found."}

        model = models[0]
        model_id = model["id"]

        # Get full context
        neighbors = self.neo4j_repo.get_neighbors(model_id, depth=1)

        # Categorize neighbors by type
        result = {
            "found": True,
            "model": model,
            "developed_by": [],
            "architecture": [],
            "techniques": [],
            "benchmarks": [],
            "datasets": [],
            "related_models": [],
            "applications": [],
        }

        for neighbor in neighbors:
            labels = neighbor.get("labels", [])
            rel_type = neighbor.get("relationship", "")

            if self._contains_any_label(labels, {"AICompany", "Organization"}):
                result["developed_by"].append(neighbor)
            elif self._has_label_from_list(labels, "Architecture"):
                result["architecture"].append(neighbor)
            elif self._has_label_from_list(labels, "Technique"):
                result["techniques"].append(neighbor)
            elif self._has_label_from_list(labels, "Benchmark"):
                result["benchmarks"].append(neighbor)
            elif self._has_label_from_list(labels, "Dataset"):
                result["datasets"].append(neighbor)
            elif self._has_label_from_list(labels, "AIModel") and rel_type in [
                "FINE_TUNED_FROM",
                "SUCCEEDED_BY",
                "SIMILAR_TO",
            ]:
                result["related_models"].append(neighbor)
            elif self._has_label_from_list(labels, "Application"):
                result["applications"].append(neighbor)

        return result

    def compare_models(self, model_a: str, model_b: str) -> Dict[str, Any]:
        """
        Compare two AI models based on their knowledge graph relationships.
        """
        knowledge_a = self.get_model_knowledge(model_a)
        knowledge_b = self.get_model_knowledge(model_b)

        if not knowledge_a.get("found") or not knowledge_b.get("found"):
            return {
                "comparable": False,
                "message": "One or both models not found in the knowledge graph.",
            }

        # Find common elements
        def get_ids(items):
            return set(item.get("id") for item in items)

        common_techniques = get_ids(knowledge_a["techniques"]) & get_ids(
            knowledge_b["techniques"]
        )
        common_benchmarks = get_ids(knowledge_a["benchmarks"]) & get_ids(
            knowledge_b["benchmarks"]
        )
        common_datasets = get_ids(knowledge_a["datasets"]) & get_ids(
            knowledge_b["datasets"]
        )

        # Check for direct connection
        connection = self.find_connections(
            knowledge_a["model"]["id"], knowledge_b["model"]["id"], max_depth=3
        )

        return {
            "comparable": True,
            "model_a": knowledge_a["model"],
            "model_b": knowledge_b["model"],
            "common_techniques": len(common_techniques),
            "common_benchmarks": len(common_benchmarks),
            "common_datasets": len(common_datasets),
            "directly_connected": connection.get("connected", False),
            "connection_path": connection.get("description")
            if connection.get("connected")
            else None,
        }

    def get_research_lineage(self, entity_id: str) -> Dict[str, Any]:
        """
        Trace the research lineage of a concept, technique, or model.

        Shows how ideas evolved through papers and implementations.
        """
        # Find papers and models that reference this entity
        cypher = """
        MATCH path = (origin {id: $entity_id})<-[*1..3]-(descendant)
        WHERE descendant:Paper OR descendant:AIModel OR descendant:Technique
        WITH path, descendant
        ORDER BY 
            CASE WHEN descendant:Paper THEN coalesce(descendant.publication_date, '1900') 
                 WHEN descendant:AIModel THEN coalesce(descendant.release_date, '1900')
                 ELSE '1900' 
            END
        RETURN DISTINCT
            descendant.id as id,
            descendant.name as name,
            labels(descendant) as labels,
            descendant.publication_date as pub_date,
            descendant.release_date as release_date
        LIMIT 20
        """

        try:
            results = self.neo4j_repo.execute_cypher(cypher, {"entity_id": entity_id})

            return {"origin": entity_id, "descendants": results, "count": len(results)}
        except Exception as e:
            return {"origin": entity_id, "descendants": [], "count": 0, "error": str(e)}

    def resolve_entity(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a name or ID to a valid entity node.
        Prioritizes: Exact ID -> Exact Name -> Fuzzy Name -> Vector Search.
        """
        if not name_or_id:
            return None

        normalized_query = self._normalize_for_match(name_or_id)
        if not normalized_query:
            return None

        # 1. Check if it's already a valid ID
        node = self.neo4j_repo.get_node_by_id(name_or_id)
        if node:
            return node

        # 2. Check for exact/fuzzy name match
        if hasattr(self.neo4j_repo, "find_node_by_name"):
            node = self.neo4j_repo.find_node_by_name(name_or_id)
            if node:
                return node

        # 3. Full-text fallback if repository supports it
        if hasattr(self.neo4j_repo, "search_nodes_by_name"):
            try:
                fulltext_results = self.neo4j_repo.search_nodes_by_name(
                    name_or_id, limit=config.ENTITY_RESOLVE_FULLTEXT_LIMIT
                )
                if fulltext_results:
                    top_fulltext = fulltext_results[0]
                    if self._token_overlap(name_or_id, top_fulltext.get("name", "")) >= 0.5:
                        return top_fulltext
            except Exception as e:
                print(f"Error during full-text entity resolution: {e}")

        # 4. Fallback to vector search with stricter acceptance checks
        try:
            results = self.neo4j_repo.vector_search(
                name_or_id,
                top_k=5,
                score_threshold=config.ENTITY_RESOLVE_MIN_VECTOR_SCORE,
            )
            if results:
                top = results[0]
                second = results[1] if len(results) > 1 else None
                if self._accept_vector_resolution(name_or_id, top, second):
                    return top
        except Exception as e:
            print(f"Error during vector search resolution: {e}")

        return None

    def _has_label(self, node: Dict[str, Any], canonical_label: str) -> bool:
        labels = node.get("labels", []) if isinstance(node, dict) else []
        return self._has_label_from_list(labels, canonical_label)

    def _has_label_from_list(self, labels: List[str], canonical_label: str) -> bool:
        target = canonical_label.lower()
        return any(str(label).lower() == target for label in labels or [])

    def _contains_any_label(self, labels: List[str], candidates: set) -> bool:
        lowered = {str(label).lower() for label in labels or []}
        return any(candidate.lower() in lowered for candidate in candidates)

    def _normalize_for_match(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
        return re.sub(r"\s+", " ", cleaned)

    def _token_overlap(self, query: str, candidate: str) -> float:
        query_tokens = set(self._normalize_for_match(query).split())
        candidate_tokens = set(self._normalize_for_match(candidate).split())
        if not query_tokens or not candidate_tokens:
            return 0.0
        common = len(query_tokens & candidate_tokens)
        return common / max(len(query_tokens), 1)

    def _accept_vector_resolution(
        self,
        query: str,
        top_result: Dict[str, Any],
        second_result: Optional[Dict[str, Any]],
    ) -> bool:
        top_score = float(top_result.get("score", 0) or 0)
        top_name = top_result.get("name") or top_result.get("id", "")
        margin = top_score - float(second_result.get("score", 0) or 0) if second_result else 1.0
        overlap = self._token_overlap(query, top_name)
        normalized_query = self._normalize_for_match(query)
        normalized_top = self._normalize_for_match(top_name)

        exact_like_match = normalized_query == normalized_top
        substring_match = (
            normalized_query in normalized_top or normalized_top in normalized_query
        ) and overlap >= config.ENTITY_RESOLVE_MIN_TOKEN_OVERLAP

        return (
            top_score >= config.ENTITY_RESOLVE_MIN_VECTOR_SCORE
            and margin >= config.ENTITY_RESOLVE_MIN_SCORE_MARGIN
            and (exact_like_match or substring_match or overlap >= 0.8)
        )

    def get_multi_entity_context(
        self, entity_ids: List[str], max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get aggregated context from multiple entities via parallel BFS.

        This is a SOTA method that:
        1. Expands subgraphs from all entities concurrently
        2. Identifies bridge nodes connecting the entities
        3. Builds relationship chain narratives

        Args:
            entity_ids: List of entity IDs to get context for
            max_depth: How far to expand from each entity

        Returns:
            {
                "entities": list of entity details,
                "bridge_nodes": nodes connecting multiple entities,
                "relationship_chains": human-readable relationship descriptions,
                "merged_context": full merged subgraph,
                "context_summary": human-readable summary
            }
        """
        if not entity_ids:
            return {
                "entities": [],
                "bridge_nodes": [],
                "relationship_chains": [],
                "merged_context": {"nodes": [], "relationships": []},
                "context_summary": "No entities provided.",
            }

        # Get individual entity details
        entities = []
        for eid in entity_ids:
            entity = self.neo4j_repo.get_node_by_id(eid)
            if entity:
                entities.append(entity)

        if not entities:
            return {
                "entities": [],
                "bridge_nodes": [],
                "relationship_chains": [],
                "merged_context": {"nodes": [], "relationships": []},
                "context_summary": "None of the provided entities were found.",
            }

        # Parallel BFS expansion from all entities
        merged = self.neo4j_repo.parallel_bfs_from_seeds(entity_ids, max_depth)
        bridge_nodes = merged.get("bridge_nodes", [])

        # Build relationship chain narratives
        relationship_chains = []
        for rel in merged.get("relationships", []):
            start = rel.get("start", "?")
            end = rel.get("end", "?")
            rel_type = rel.get("type", "RELATED_TO")
            chain = f"{start} --[{rel_type}]--> {end}"
            relationship_chains.append(chain)

        # Build context summary
        entity_names = [e.get("name", e.get("id", "?")) for e in entities]
        summary_parts = [
            f"Context for {len(entities)} entities: {', '.join(entity_names)}",
        ]

        if bridge_nodes:
            bridge_names = [b.get("name", b.get("id", "?")) for b in bridge_nodes[:5]]
            summary_parts.append(
                f"Bridge nodes connecting them: {', '.join(bridge_names)}"
            )

        total_nodes = len(merged.get("nodes", []))
        total_rels = len(merged.get("relationships", []))
        summary_parts.append(
            f"Expanded subgraph contains {total_nodes} nodes and {total_rels} relationships."
        )

        return {
            "entities": entities,
            "bridge_nodes": bridge_nodes,
            "relationship_chains": relationship_chains[:20],
            "merged_context": {
                "nodes": merged.get("nodes", []),
                "relationships": merged.get("relationships", []),
            },
            "context_summary": "\n".join(summary_parts),
        }
