"""
GraphController - Orchestrates graph building with batch processing and checkpointing.
"""

import os
from datetime import datetime
from typing import Optional

from src.services.ingest_service import IngestService
from src.services.graph_service import GraphService
from src.services.progress_tracker import ChunkProgressTracker
from src.repositories.file_repository import FileRepository
from src.repositories.neo4j_repository import Neo4jRepository
from src.repositories.processing_state import ProcessingState, ProcessingStateRepository
from src import config


class GraphController:
    """Controller for graph building operations with large-scale processing support."""

    def __init__(self):
        # Wire up dependencies (following DDD architecture)
        file_repo = FileRepository()
        neo4j_repo = Neo4jRepository()

        self.ingest_service = IngestService(file_repo)
        self.graph_service = GraphService(neo4j_repo)
        self.file_repo = file_repo
        self.neo4j_repo = neo4j_repo
        self.state_repo = ProcessingStateRepository()

    # =========================================================================
    # LEGACY METHODS (for backward compatibility)
    # =========================================================================

    def process_folder(self, folder_path="./data"):
        """Legacy method: Validates input and orchestrates graph building from folder."""
        try:
            docs = self.ingest_service.load_documents_from_folder(folder_path)
            if not docs:
                return "No documents found to process."

            self.graph_service.build_graph(docs)
            return "Successfully built graph from folder."
        except Exception as e:
            return f"Error during graph building: {str(e)}"

    def chat(self, question: str):
        """Orchestrates graph querying."""
        if not question.strip():
            return "Question cannot be empty."

        try:
            return self.graph_service.query_graph(question)
        except Exception as e:
            return f"Error during chat: {str(e)}"

    # =========================================================================
    # LARGE-SCALE PROCESSING METHODS
    # =========================================================================

    def process_folder_batch(
        self, folder_path="./data", resume: bool = True, job_id: str = "default"
    ) -> str:
        """
        Process folder with batching, checkpointing, and progress tracking.

        Args:
            folder_path: Path to folder containing PDF/TXT files
            resume: If True, resume from last checkpoint
            job_id: Unique identifier for this processing job

        Returns:
            Summary message
        """
        print("\n" + "=" * 60)
        print("🚀 LARGE-SCALE GRAPH BUILDER")
        print("=" * 60)

        # Step 1: Get folder statistics
        print(f"\n📂 Scanning '{folder_path}'...")
        stats = self.ingest_service.get_folder_statistics(folder_path)

        if stats["total_files"] == 0:
            return "No PDF or TXT files found in the folder."

        print(f"📊 Found {stats['total_files']} files ({stats['total_size_mb']} MB)")
        print(f"📝 Estimated chunks: ~{stats['estimated_chunks']:,}")

        # Step 2: Load or create processing state
        state = None
        if resume:
            state = self.state_repo.load_checkpoint(job_id)

        if state is None:
            state = ProcessingState(
                total_files=stats["total_files"], start_time=datetime.now().isoformat()
            )
            print("🆕 Starting new processing job")
        else:
            print(
                f"♻️ Resuming from checkpoint ({state.processed_count}/{state.total_files} files done)"
            )

        # Step 3: Prepare database
        self.graph_service.prepare_database()

        # Step 4: Get pending files
        all_files = stats["files"]
        pending_files = self.state_repo.get_pending_files(state, all_files)

        if not pending_files:
            return self.state_repo.generate_summary(state)

        print(f"📋 Files to process: {len(pending_files)}")

        # Step 5: Initialize progress tracker
        progress = ChunkProgressTracker(
            total_files=len(pending_files),
            log_file=os.path.join(config.LOG_DIR, f"processing_{job_id}.log")
            if config.ENABLE_DETAILED_LOGGING
            else None,
        )

        # Step 6: Process files in batches
        files_since_checkpoint = 0

        try:
            for file_path in pending_files:
                state.current_file = file_path
                progress.start_item(os.path.basename(file_path))

                try:
                    # Load and chunk file
                    chunks, error = self.ingest_service.load_and_chunk_file(file_path)

                    if error:
                        self.state_repo.mark_file_failed(state, file_path, error)
                        progress.fail_item(error=error)
                        continue

                    if not chunks:
                        self.state_repo.mark_file_skipped(state, file_path)
                        progress.complete_item()
                        continue

                    progress.set_file_chunks(len(chunks))

                    # Build graph from chunks
                    result = self.graph_service.build_graph_batch(
                        chunks,
                        progress_callback=lambda p, t: setattr(
                            progress,
                            "completed_chunks",
                            progress.completed_chunks
                            + (p - (getattr(progress, "_last_p", 0))),
                        )
                        or setattr(progress, "_last_p", p),
                    )
                    if hasattr(progress, "_last_p"):
                        delattr(progress, "_last_p")

                    # Mark complete and update state
                    self.state_repo.mark_file_complete(
                        state,
                        file_path,
                        chunks_processed=result.get("chunks_processed", len(chunks)),
                        nodes_created=result.get("nodes_created", 0),
                        relationships_created=result.get("relationships_created", 0),
                    )

                    progress.complete_item()
                    progress.completed_chunks += result.get(
                        "chunks_processed", len(chunks)
                    )

                except Exception as e:
                    self.state_repo.mark_file_failed(state, file_path, str(e))
                    progress.fail_item(error=str(e))

                # Checkpoint periodically
                files_since_checkpoint += 1
                if files_since_checkpoint >= config.CHECKPOINT_INTERVAL:
                    self.state_repo.save_checkpoint(state, job_id)
                    files_since_checkpoint = 0

        except KeyboardInterrupt:
            print("\n\n⚠️ Processing interrupted! Saving checkpoint...")
            self.state_repo.save_checkpoint(state, job_id)
            print("✅ Checkpoint saved. Run again to resume.")
            raise

        # Final checkpoint
        self.state_repo.save_checkpoint(state, job_id)

        # Display summary
        progress.display_summary()

        return self.state_repo.generate_summary(state)

    def get_processing_status(self, job_id: str = "default") -> dict:
        """Get current processing status for a job."""
        state = self.state_repo.load_checkpoint(job_id)

        if state is None:
            return {"status": "no_job", "message": "No processing job found"}

        return {
            "status": "in_progress" if state.current_file else "completed",
            "total_files": state.total_files,
            "processed_files": state.processed_count,
            "failed_files": state.failed_count,
            "progress_percentage": state.progress_percentage,
            "current_file": state.current_file,
            "start_time": state.start_time,
            "last_checkpoint": state.last_checkpoint_time,
            "total_chunks": state.total_chunks_processed,
            "total_nodes": state.total_nodes_created,
            "total_relationships": state.total_relationships_created,
        }

    def retry_failed_files(self, job_id: str = "default") -> str:
        """Reprocess files that previously failed."""
        state = self.state_repo.load_checkpoint(job_id)

        if state is None:
            return "No processing job found to retry."

        failed_files = self.state_repo.get_failed_files(state)

        if not failed_files:
            return "No failed files to retry."

        print(f"\n🔄 Retrying {len(failed_files)} failed files...")

        # Clear failed status for retry
        for file_path in failed_files:
            del state.failed_files[file_path]

        # Save state
        self.state_repo.save_checkpoint(state, job_id)

        # Resume processing (will pick up the previously failed files)
        return self.process_folder_batch(resume=True, job_id=job_id)

    def reset_job(self, job_id: str = "default") -> str:
        """Reset a processing job (start fresh)."""
        deleted = self.state_repo.delete_checkpoint(job_id)

        if deleted:
            return f"✅ Job '{job_id}' reset. Will start fresh on next run."
        else:
            return f"No job '{job_id}' found to reset."

    def get_database_stats(self) -> dict:
        """Get Neo4j database statistics."""
        return self.neo4j_repo.get_statistics()

    def preview_folder(self, folder_path="./data") -> str:
        """Preview what would be processed without actually processing."""
        stats = self.ingest_service.get_folder_statistics(folder_path)

        lines = [
            "\n" + "=" * 60,
            "📊 FOLDER PREVIEW",
            "=" * 60,
            f"📂 Path: {os.path.abspath(folder_path)}",
            f"📄 PDF Files: {stats['pdf_count']}",
            f"📝 TXT Files: {stats['txt_count']}",
            f"📊 Total Files: {stats['total_files']}",
            f"💾 Total Size: {stats['total_size_mb']} MB",
            f"📝 Estimated Chunks: ~{stats['estimated_chunks']:,}",
            "",
            "📋 Sample files:",
        ]

        # Show first 10 files
        for f in stats["files"][:10]:
            lines.append(f"   - {os.path.basename(f)}")

        if len(stats["files"]) > 10:
            lines.append(f"   ... and {len(stats['files']) - 10} more")

        lines.append("=" * 60)

        return "\n".join(lines)
