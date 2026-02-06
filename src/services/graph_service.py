"""
GraphService - Handles graph extraction with concurrent processing and rate limiting.
"""

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
                request_timeout=120,  # Longer timeout for large chunks
            )
        return self._llm

    @property
    def transformer(self) -> LLMGraphTransformer:
        """Lazy initialization of graph transformer."""
        if self._transformer is None:
            self._transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=config.ALLOWED_NODES,
                allowed_relationships=config.ALLOWED_RELATIONSHIPS,
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

        # Enrich relationships with source text for better visualization
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

        Args:
            docs: List of document chunks to process
            batch_size: Number of chunks per batch
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            Statistics about the processing
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
                estimated_tokens = len(batch) * 2000  # Rough estimate
                wait_time = self.rate_limiter.acquire(estimated_tokens)

                if wait_time > 1:
                    print(f"⏳ Rate limited, waited {wait_time:.1f}s")

                # Process batch
                graph_docs = self._process_batch_with_retry(batch)

                # Enrich relationships with source text for better visualization
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
                all_graph_docs, batch_size=100, include_source=True
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

        Args:
            docs: List of document chunks to process
            max_workers: Maximum concurrent processing threads
            progress_callback: Optional callback for progress updates

        Returns:
            Processing statistics
        """
        max_workers = max_workers or config.MAX_CONCURRENT_LLM_CALLS
        batch_size = config.BATCH_SIZE_CHUNKS

        if not docs:
            return {"chunks_processed": 0, "graph_documents": 0}

        # Create batches
        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

        all_graph_docs = []
        processed_batches = 0
        errors = 0

        print(
            f"🚀 Processing {len(docs)} chunks in {len(batches)} batches with {max_workers} workers..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_single_batch, batch): idx
                for idx, batch in enumerate(batches)
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]

                try:
                    graph_docs = future.result()

                    # Enrich relationships
                    self._enrich_graph_documents(graph_docs)

                    all_graph_docs.extend(graph_docs)
                    processed_batches += 1

                    if progress_callback:
                        progress_callback(processed_batches, len(batches))

                except Exception as e:
                    print(f"⚠️ Batch {batch_idx} failed: {e}")
                    errors += 1

        # Save all results to Neo4j
        if all_graph_docs:
            print(f"💾 Saving {len(all_graph_docs)} graph documents to Neo4j...")
            stats = self.neo4j_repo.add_graph_documents_batch(
                all_graph_docs, batch_size=100, include_source=True
            )
        else:
            stats = {"total_nodes": 0, "total_relationships": 0}

        processed_chunks = sum(len(batches[i]) for i in range(processed_batches))

        return {
            "chunks_processed": processed_chunks,
            "batches_processed": processed_batches,
            "batches_failed": errors,
            "graph_documents": len(all_graph_docs),
            "nodes_created": stats.get("total_nodes", 0),
            "relationships_created": stats.get("total_relationships", 0),
        }

    def _process_single_batch(self, batch: List[Document]) -> List:
        """Process a single batch with rate limiting."""
        # Acquire rate limit permission
        estimated_tokens = len(batch) * 2000
        self.rate_limiter.acquire(estimated_tokens)

        # Process with retry
        return self._process_batch_with_retry(batch)

    def query_graph(self, question: str) -> str:
        """Queries the knowledge graph."""
        return self.neo4j_repo.query(question)

    def prepare_database(self):
        """Prepare database for large-scale ingestion."""
        print("📇 Preparing database indexes...")
        self.neo4j_repo.create_indexes()
        print("✅ Database ready for ingestion")

    def _enrich_graph_documents(self, graph_documents: List):
        """Adds source text and metadata to extracted relationships for better visualization."""
        for g_doc in graph_documents:
            source_text = (
                g_doc.source.page_content
                if hasattr(g_doc, "source") and hasattr(g_doc.source, "page_content")
                else ""
            )
            source_id = (
                g_doc.source.metadata.get("source", "unknown")
                if hasattr(g_doc, "source") and hasattr(g_doc.source, "metadata")
                else "unknown"
            )

            for rel in g_doc.relationships:
                # Add source text and metadata to the relationship itself
                # In the Neo4j Browser, clicking the relationship will now show these properties
                rel.properties["source_chunk_text"] = source_text
                rel.properties["source_file"] = source_id
