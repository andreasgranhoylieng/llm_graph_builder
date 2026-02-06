"""
TraversalService - Advanced graph traversal and context extraction for RAG.
"""

from typing import List, Dict, Any
from src.services.interfaces import INeo4jRepository, ITraversalService


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

        if path:
            # Build human-readable description
            description_parts = []
            nodes = path.get("nodes", [])
            relationships = path.get("relationships", [])

            for i, node in enumerate(nodes):
                description_parts.append(node.get("name", node.get("id")))
                if i < len(relationships):
                    rel_type = relationships[i].get("type", "RELATED_TO")
                    description_parts.append(f" --[{rel_type}]--> ")

            return {
                "connected": True,
                "path": path,
                "hops": len(nodes) - 1,
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

            if "AICompany" in labels or "Organization" in labels:
                result["developed_by"].append(neighbor)
            elif "Architecture" in labels:
                result["architecture"].append(neighbor)
            elif "Technique" in labels:
                result["techniques"].append(neighbor)
            elif "Benchmark" in labels:
                result["benchmarks"].append(neighbor)
            elif "Dataset" in labels:
                result["datasets"].append(neighbor)
            elif "AIModel" in labels and rel_type in [
                "FINE_TUNED_FROM",
                "SUCCEEDED_BY",
                "SIMILAR_TO",
            ]:
                result["related_models"].append(neighbor)
            elif "Application" in labels:
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
