"""
GraphService - Handles graph extraction with concurrent processing and rate limiting.
Enhanced with description extraction for better embeddings.
"""

import math
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.services.interfaces import INeo4jRepository
from src.services.rate_limiter import get_rate_limiter
from src import config


class GraphService:
    """Service for extracting graph data from documents with concurrent processing."""

    def __init__(self, neo4j_repo: INeo4jRepository):
        self.neo4j_repo = neo4j_repo
        self._llm = None
        self._transformer = None
        self.rate_limiter = get_rate_limiter()

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                temperature=config.LLM_TEMPERATURE,
                model_name=config.LLM_MODEL,
                request_timeout=120,
            )
        return self._llm

    @property
    def transformer(self) -> LLMGraphTransformer:
        """Lazy initialization of graph transformer with Generative AI schema."""
        if self._transformer is None:
            self._transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=config.ALLOWED_NODES,
                allowed_relationships=config.ALLOWED_RELATIONSHIPS,
                node_properties=config.GRAPH_EXTRACT_NODE_PROPERTIES,
                relationship_properties=config.GRAPH_EXTRACT_RELATIONSHIP_PROPERTIES,
            )
        return self._transformer

    def build_graph(self, docs: List[Document]):
        """
        Legacy method: Extracts entities and relationships and saves them to Neo4j.
        For large-scale processing, use build_graph_batch.
        """
        if not docs:
            print("⚠️ No documents to process.")
            return

        print("🚀 Initializing LLM and Graph Transformer...")
        print("🧠 Extracting graph data (this may take a moment)...")

        graph_documents = self.transformer.convert_to_graph_documents(docs)

        # Enrich with descriptions and source tracking
        self._enrich_graph_documents(graph_documents)

        print(f"✅ Extracted {len(graph_documents)} graph components.")
        print("💾 Saving to Neo4j...")

        self.neo4j_repo.add_graph_documents(graph_documents, include_source=True)

        print("🎉 Done! Data is now in Neo4j.")

    def build_graph_batch(
        self,
        docs: List[Document],
        batch_size: int = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Process documents in batches with rate limiting.
        """
        batch_size = batch_size or config.BATCH_SIZE_CHUNKS

        if not docs:
            return {"chunks_processed": 0, "graph_documents": 0}

        total_chunks = len(docs)
        all_graph_docs = []
        processed = 0
        errors = 0

        print(f"🧠 Processing {total_chunks} chunks in batches of {batch_size}...")

        for i in range(0, total_chunks, batch_size):
            batch = docs[i : i + batch_size]

            try:
                # Acquire rate limit permission
                estimated_tokens = len(batch) * 2000
                wait_time = self.rate_limiter.acquire(estimated_tokens)

                if wait_time > 1:
                    print(f"⏳ Rate limited, waited {wait_time:.1f}s")

                # Process batch
                graph_docs = self._process_batch_with_retry(batch)

                # Enrich with descriptions and source tracking
                self._enrich_graph_documents(graph_docs)

                all_graph_docs.extend(graph_docs)
                processed += len(batch)

                if progress_callback:
                    progress_callback(processed, total_chunks)

            except Exception as e:
                print(f"⚠️ Batch {i // batch_size} failed: {e}")
                errors += len(batch)
                continue

        # Save to Neo4j
        if all_graph_docs:
            print(f"💾 Saving {len(all_graph_docs)} graph documents to Neo4j...")
            stats = self.neo4j_repo.add_graph_documents_batch(
                all_graph_docs,
                batch_size=config.NEO4J_INGEST_BATCH_SIZE,
                include_source=True,
            )
        else:
            stats = {"total_nodes": 0, "total_relationships": 0}

        return {
            "chunks_processed": processed,
            "chunks_failed": errors,
            "graph_documents": len(all_graph_docs),
            "nodes_created": stats.get("total_nodes", 0),
            "relationships_created": stats.get("total_relationships", 0),
        }

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=config.RETRY_DELAY_SECONDS, min=2, max=60),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _process_batch_with_retry(self, batch: List[Document]) -> List:
        """Process a batch with automatic retry on transient failures."""
        return self.transformer.convert_to_graph_documents(batch)

    def build_graph_concurrent(
        self,
        docs: List[Document],
        max_workers: int = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Process documents using concurrent threads for higher throughput.
        """
        max_workers = max_workers or config.MAX_CONCURRENT_LLM_CALLS
        batch_size = self._resolve_concurrent_batch_size(len(docs), max_workers)

        if not docs:
            return {"chunks_processed": 0, "graph_documents": 0}

        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

        write_buffer: List = []
        processed_chunks = 0
        failed_chunks = 0

        graph_documents_count = 0
        total_nodes = 0
        total_relationships = 0
        flush_size = max(1, config.GRAPH_WRITE_FLUSH_SIZE)

        print(
            f"Processing {len(docs)} chunks in {len(batches)} batches with {max_workers} workers..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_single_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                batch_len = len(batch)

                try:
                    graph_docs = future.result()

                    if graph_docs:
                        self._enrich_graph_documents(graph_docs)
                        graph_documents_count += len(graph_docs)
                        write_buffer.extend(graph_docs)

                        if len(write_buffer) >= flush_size:
                            stats = self._flush_graph_documents(write_buffer)
                            total_nodes += stats.get("total_nodes", 0)
                            total_relationships += stats.get("total_relationships", 0)

                    processed_chunks += batch_len

                except Exception as e:
                    print(f"Batch failed: {e}")
                    failed_chunks += batch_len

                if progress_callback:
                    progress_callback(processed_chunks + failed_chunks, len(docs))

        if write_buffer:
            stats = self._flush_graph_documents(write_buffer)
            total_nodes += stats.get("total_nodes", 0)
            total_relationships += stats.get("total_relationships", 0)

        return {
            "chunks_processed": processed_chunks,
            "chunks_failed": failed_chunks,
            "batches_processed": len(batches),
            "graph_documents": graph_documents_count,
            "nodes_created": total_nodes,
            "relationships_created": total_relationships,
        }

    def _process_single_batch(self, batch: List[Document]) -> List:
        """Process a single batch with rate limiting."""
        estimated_tokens = len(batch) * 2000
        wait_time = self.rate_limiter.acquire(estimated_tokens)
        if config.GRAPH_LOG_RATE_LIMIT_WAIT and wait_time > 1:
            print(
                f"Rate limiter waited {wait_time:.1f}s for batch size {len(batch)} "
                f"(estimated tokens={estimated_tokens})"
            )
        return self._process_batch_with_retry(batch)

    def _resolve_concurrent_batch_size(self, total_docs: int, max_workers: int) -> int:
        """
        Resolve an adaptive batch size so smaller files can still use concurrency.

        For small files, fixed large batches (e.g. 50) collapse to 1-2 batches and
        run mostly serially. We target multiple batches per worker, then cap by the
        configured maximum batch size.
        """
        configured_max = max(1, config.BATCH_SIZE_CHUNKS or 10)
        if total_docs <= 0:
            return configured_max

        target_batches = max(
            1,
            max_workers * max(1, config.GRAPH_BATCHES_PER_WORKER),
            config.GRAPH_MIN_BATCHES_PER_FILE,
        )
        adaptive_size = max(1, math.ceil(total_docs / target_batches))
        return min(configured_max, adaptive_size)

    def _flush_graph_documents(self, graph_docs: List) -> dict:
        """Persist queued graph documents in larger writes for better throughput."""
        if not graph_docs:
            return {"total_nodes": 0, "total_relationships": 0}

        payload = list(graph_docs)
        graph_docs.clear()
        print(f"Saving {len(payload)} graph documents to Neo4j...")
        return self.neo4j_repo.add_graph_documents_batch(
            payload,
            batch_size=config.NEO4J_INGEST_BATCH_SIZE,
            include_source=True,
        )

    def query_graph(self, question: str) -> str:
        """Queries the knowledge graph using Cypher (legacy method)."""
        return self.neo4j_repo.query(question)

    def query_graph_hybrid(self, question: str) -> dict:
        """
        Queries the knowledge graph using hybrid vector + graph approach.

        Returns:
            {
                "answer": str,
                "entities_found": List,
                "graph_context": List,
                "confidence": float
            }
        """
        return self.neo4j_repo.query_hybrid(question)

    def prepare_database(self):
        """Prepare database for large-scale ingestion."""
        print("📇 Preparing database indexes...")
        self.neo4j_repo.create_indexes()
        print("✅ Database ready for ingestion")

    def _enrich_graph_documents(self, graph_documents: List):
        """
        Enriches extracted graph documents with:
        1. Source text and metadata on relationships
        2. Description generation for nodes without descriptions
        3. Confidence scores
        """
        for g_doc in graph_documents:
            # Extract source info
            source_text = ""
            source_id = "unknown"

            if hasattr(g_doc, "source") and g_doc.source:
                if hasattr(g_doc.source, "page_content"):
                    source_text = g_doc.source.page_content
                if hasattr(g_doc.source, "metadata"):
                    source_id = g_doc.source.metadata.get("source", "unknown")

            # Enrich nodes with description if missing
            for node in g_doc.nodes:
                if not hasattr(node, "properties"):
                    node.properties = {}

                # Add source tracking
                node.properties["source_document"] = source_id

                # Clean and set proper name
                clean_name = self._clean_node_name(node.id)
                node.properties["name"] = clean_name

                # Generate description from source if not present
                if not node.properties.get("description"):
                    # Extract relevant context from source text
                    description = self._extract_entity_description(node.id, source_text)
                    if description:
                        node.properties["description"] = description

                # Add confidence score (could be enhanced with actual confidence)
                if "confidence" not in node.properties:
                    node.properties["confidence"] = 0.8

            # Enrich relationships
            for rel in g_doc.relationships:
                if not hasattr(rel, "properties"):
                    rel.properties = {}

                rel.properties["source_chunk_text"] = (
                    source_text[:500] if source_text else ""
                )
                rel.properties["source_file"] = source_id
                rel.properties["extracted_at"] = self._get_timestamp()

    def _extract_entity_description(
        self, entity_name: str, source_text: str, max_length: int = 200
    ) -> str:
        """
        Extract a description for an entity from the source text.

        Uses simple heuristics:
        1. Find sentences containing the entity name
        2. Return the most relevant sentence as description
        """
        if not source_text or not entity_name:
            return ""

        # Normalize for matching
        entity_lower = entity_name.lower()

        # Split into sentences
        sentences = source_text.replace("\n", " ").split(".")

        # Find sentences mentioning the entity
        relevant_sentences = []
        for sentence in sentences:
            if entity_lower in sentence.lower():
                cleaned = sentence.strip()
                if len(cleaned) > 20:  # Skip very short fragments
                    relevant_sentences.append(cleaned)

        if not relevant_sentences:
            return ""

        # Return the first/most relevant sentence, truncated
        description = relevant_sentences[0]
        if len(description) > max_length:
            description = description[:max_length].rsplit(" ", 1)[0] + "..."

        return description

    def _clean_node_name(self, node_id: str, max_length: int = 50) -> str:
        """
        Clean and normalize a node ID to create a readable name.

        Handles:
        - URLs: Extract domain or title
        - Long paper titles: Truncate to first meaningful part
        - Numeric IDs: Convert to readable format
        - Normal names: Title case normalization
        """
        import re

        if not node_id:
            return "Unknown"

        name = str(node_id).strip()

        # Skip if it's a URL - try to extract meaningful part
        if name.startswith("http://") or name.startswith("https://"):
            # Extract the last path segment or domain
            parts = name.split("/")
            for part in reversed(parts):
                if part and not part.startswith("http") and len(part) > 3:
                    name = part.replace("-", " ").replace("_", " ")
                    break
            else:
                return "External Link"

        # Remove common URL artifacts
        name = re.sub(r"https?://[^\s]+", "", name)
        name = re.sub(r"www\.\S+", "", name)

        # If it looks like a long sentence/title, truncate smartly
        if len(name) > max_length:
            # Try to find a good break point
            if ":" in name[:max_length]:
                name = name.split(":")[0].strip()
            elif " - " in name[:max_length]:
                name = name.split(" - ")[0].strip()
            elif "'" in name[:max_length]:
                # Preserve quoted names like "Musk's xAI"
                pass
            else:
                # Just truncate at word boundary
                name = name[:max_length].rsplit(" ", 1)[0]
                if not name.endswith("..."):
                    name += "..."

        # Clean up extra whitespace
        name = " ".join(name.split())

        return name if name else "Unknown"

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat()

    def get_extraction_stats(self) -> dict:
        """Get statistics about the graph extraction process."""
        return {
            "rate_limiter": self.rate_limiter.get_stats() if self.rate_limiter else {}
        }

