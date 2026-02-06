"""
ProcessingState Repository - Handles checkpointing and state persistence for large-scale document processing.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set
from src import config


@dataclass
class ProcessingState:
    """Tracks the state of a batch processing job."""

    # File tracking
    processed_files: List[str] = field(default_factory=list)
    failed_files: Dict[str, str] = field(
        default_factory=dict
    )  # {filepath: error_message}
    skipped_files: List[str] = field(default_factory=list)

    # Progress statistics
    total_files: int = 0
    total_chunks_processed: int = 0
    total_nodes_created: int = 0
    total_relationships_created: int = 0

    # Timing
    start_time: Optional[str] = None
    last_checkpoint_time: Optional[str] = None

    # Current batch tracking
    current_batch_index: int = 0
    current_file: Optional[str] = None

    @property
    def processed_count(self) -> int:
        return len(self.processed_files)

    @property
    def failed_count(self) -> int:
        return len(self.failed_files)

    @property
    def progress_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_count / self.total_files) * 100


class ProcessingStateRepository:
    """Handles persistence of processing state for checkpointing and resume."""

    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        """Create checkpoint directory if it doesn't exist."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _get_checkpoint_path(self, job_id: str = "default") -> str:
        """Get the path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_{job_id}.json")

    def save_checkpoint(self, state: ProcessingState, job_id: str = "default") -> None:
        """Persist state to disk."""
        state.last_checkpoint_time = datetime.now().isoformat()

        checkpoint_path = self._get_checkpoint_path(job_id)
        temp_path = checkpoint_path + ".tmp"

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2)

            # Atomic rename for safety
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            os.rename(temp_path, checkpoint_path)

            print(
                f"💾 Checkpoint saved: {state.processed_count}/{state.total_files} files processed"
            )
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_checkpoint(self, job_id: str = "default") -> Optional[ProcessingState]:
        """Load state from disk if it exists."""
        checkpoint_path = self._get_checkpoint_path(job_id)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = ProcessingState(**data)
            print(
                f"📂 Loaded checkpoint: {state.processed_count}/{state.total_files} files already processed"
            )
            return state
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, job_id: str = "default") -> bool:
        """Delete a checkpoint file."""
        checkpoint_path = self._get_checkpoint_path(job_id)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            return True
        return False

    def mark_file_complete(
        self,
        state: ProcessingState,
        file_path: str,
        chunks_processed: int = 0,
        nodes_created: int = 0,
        relationships_created: int = 0,
    ) -> None:
        """Mark a file as successfully processed."""
        if file_path not in state.processed_files:
            state.processed_files.append(file_path)

        # Remove from failed if it was previously failed
        if file_path in state.failed_files:
            del state.failed_files[file_path]

        state.total_chunks_processed += chunks_processed
        state.total_nodes_created += nodes_created
        state.total_relationships_created += relationships_created
        state.current_file = None

    def mark_file_failed(
        self, state: ProcessingState, file_path: str, error_message: str
    ) -> None:
        """Record a file as failed for later retry."""
        state.failed_files[file_path] = error_message
        state.current_file = None
        print(f"❌ Failed: {os.path.basename(file_path)} - {error_message}")

    def mark_file_skipped(self, state: ProcessingState, file_path: str) -> None:
        """Mark a file as skipped (e.g., already processed)."""
        if file_path not in state.skipped_files:
            state.skipped_files.append(file_path)

    def get_pending_files(
        self, state: ProcessingState, all_files: List[str]
    ) -> List[str]:
        """Get files that haven't been processed yet."""
        processed_set = set(state.processed_files)
        skipped_set = set(state.skipped_files)

        return [f for f in all_files if f not in processed_set and f not in skipped_set]

    def get_failed_files(self, state: ProcessingState) -> List[str]:
        """Get list of files that failed processing."""
        return list(state.failed_files.keys())

    def generate_summary(self, state: ProcessingState) -> str:
        """Generate a human-readable summary of the processing state."""
        lines = [
            "\n" + "=" * 60,
            "📊 PROCESSING SUMMARY",
            "=" * 60,
            f"Total Files: {state.total_files}",
            f"✅ Processed: {state.processed_count}",
            f"❌ Failed: {state.failed_count}",
            f"⏭️  Skipped: {len(state.skipped_files)}",
            f"📊 Progress: {state.progress_percentage:.1f}%",
            "",
            f"📝 Chunks Processed: {state.total_chunks_processed:,}",
            f"🔵 Nodes Created: {state.total_nodes_created:,}",
            f"🔗 Relationships Created: {state.total_relationships_created:,}",
            "",
            f"⏱️  Started: {state.start_time or 'N/A'}",
            f"💾 Last Checkpoint: {state.last_checkpoint_time or 'N/A'}",
        ]

        if state.failed_files:
            lines.append("\n❌ FAILED FILES:")
            for filepath, error in list(state.failed_files.items())[:10]:
                lines.append(f"   - {os.path.basename(filepath)}: {error[:50]}...")
            if len(state.failed_files) > 10:
                lines.append(f"   ... and {len(state.failed_files) - 10} more")

        lines.append("=" * 60 + "\n")

        return "\n".join(lines)
