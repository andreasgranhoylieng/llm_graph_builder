"""
ProgressTracker Service - Real-time progress monitoring for large-scale processing.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List
from collections import deque


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""

    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)

    # Rolling average for ETA calculation
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def progress_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def average_duration(self) -> float:
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)

    @property
    def eta_seconds(self) -> float:
        remaining = self.total_items - self.completed_items - self.failed_items
        if remaining <= 0 or self.average_duration == 0:
            return 0.0
        return remaining * self.average_duration


class ProgressTracker:
    """
    Tracks and displays progress for long-running operations.
    Provides ETA, progress bars, and detailed statistics.
    """

    def __init__(
        self, total_items: int, description: str = "Processing", log_file: str = None
    ):
        self.stats = ProgressStats(total_items=total_items)
        self.description = description
        self.log_file = log_file

        self._last_item_start: float = 0
        self._current_item: str = ""
        self._callbacks: List[Callable] = []

        # For log file
        if self.log_file:
            self._init_log_file()

    def _init_log_file(self):
        """Initialize the log file."""
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Processing Log Started: {datetime.now().isoformat()} ===\n")
            f.write(f"Total items: {self.stats.total_items}\n\n")

    def _log(self, message: str):
        """Write to log file if configured."""
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def start_item(self, item_name: str):
        """Mark the start of processing an item."""
        self._last_item_start = time.time()
        self._current_item = item_name

    def complete_item(self, item_name: str = None):
        """Mark an item as completed."""
        duration = time.time() - self._last_item_start
        self.stats.recent_durations.append(duration)
        self.stats.completed_items += 1

        item = item_name or self._current_item
        self._log(f"✅ Completed: {item} ({duration:.2f}s)")
        self._update_display()

        for callback in self._callbacks:
            callback(self.stats)

    def fail_item(self, item_name: str = None, error: str = ""):
        """Mark an item as failed."""
        self.stats.failed_items += 1

        item = item_name or self._current_item
        self._log(f"❌ Failed: {item} - {error}")
        self._update_display()

    def update_total(self, new_total: int):
        """Update the total item count (if discovered during processing)."""
        self.stats.total_items = new_total

    def add_callback(self, callback: Callable):
        """Add a callback to be called on progress updates."""
        self._callbacks.append(callback)

    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _create_progress_bar(self, width: int = 30) -> str:
        """Create a visual progress bar."""
        filled = int(width * self.stats.progress_percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _update_display(self):
        """Update the console display."""
        stats = self.stats

        # Build progress line
        bar = self._create_progress_bar()
        percentage = f"{stats.progress_percentage:5.1f}%"
        counts = f"{stats.completed_items}/{stats.total_items}"
        elapsed = self._format_time(stats.elapsed_seconds)
        eta = (
            self._format_time(stats.eta_seconds)
            if stats.eta_seconds > 0
            else "calculating..."
        )

        line = f"\r{self.description}: {bar} {percentage} ({counts}) | Elapsed: {elapsed} | ETA: {eta}"

        if stats.failed_items > 0:
            line += f" | ❌ {stats.failed_items} failed"

        # Pad to overwrite previous line completely
        line = line.ljust(120)

        sys.stdout.write(line)
        sys.stdout.flush()

    def display_summary(self):
        """Display final summary."""
        stats = self.stats

        print("\n")
        print("=" * 60)
        print(f"📊 {self.description.upper()} - COMPLETE")
        print("=" * 60)
        print(f"✅ Completed: {stats.completed_items}")
        print(f"❌ Failed: {stats.failed_items}")
        print(
            f"📊 Success Rate: {(stats.completed_items / max(stats.total_items, 1)) * 100:.1f}%"
        )
        print(f"⏱️  Total Time: {self._format_time(stats.elapsed_seconds)}")

        if stats.completed_items > 0:
            avg_time = stats.elapsed_seconds / stats.completed_items
            print(f"⚡ Avg Time per Item: {avg_time:.2f}s")

        print("=" * 60)

        self._log("\n=== Processing Complete ===")
        self._log(f"Completed: {stats.completed_items}, Failed: {stats.failed_items}")
        self._log(f"Total time: {self._format_time(stats.elapsed_seconds)}")

    def get_stats(self) -> dict:
        """Get current statistics as a dictionary."""
        return {
            "total": self.stats.total_items,
            "completed": self.stats.completed_items,
            "failed": self.stats.failed_items,
            "progress_percentage": round(self.stats.progress_percentage, 2),
            "elapsed_seconds": round(self.stats.elapsed_seconds, 2),
            "eta_seconds": round(self.stats.eta_seconds, 2),
            "average_duration": round(self.stats.average_duration, 2),
        }


class ChunkProgressTracker(ProgressTracker):
    """
    Specialized progress tracker for document chunk processing.
    Tracks both files and chunks.
    """

    def __init__(self, total_files: int, log_file: str = None):
        super().__init__(total_files, "Files", log_file)

        self.total_chunks = 0
        self.completed_chunks = 0
        self.current_file_chunks = 0

    def set_file_chunks(self, chunk_count: int):
        """Set the number of chunks for the current file."""
        self.current_file_chunks = chunk_count
        self.total_chunks += chunk_count

    def complete_chunk(self):
        """Mark a chunk as completed."""
        self.completed_chunks += 1

    def _update_display(self):
        """Override to show both file and chunk progress."""
        stats = self.stats

        bar = self._create_progress_bar()
        percentage = f"{stats.progress_percentage:5.1f}%"
        file_counts = f"{stats.completed_items}/{stats.total_items} files"
        chunk_counts = f"{self.completed_chunks:,} chunks"
        elapsed = self._format_time(stats.elapsed_seconds)
        eta = (
            self._format_time(stats.eta_seconds)
            if stats.eta_seconds > 0
            else "calculating..."
        )

        line = f"\r{bar} {percentage} | {file_counts} | {chunk_counts} | Elapsed: {elapsed} | ETA: {eta}"

        if stats.failed_items > 0:
            line += f" | ❌ {stats.failed_items}"

        line = line.ljust(130)

        sys.stdout.write(line)
        sys.stdout.flush()

    def get_stats(self) -> dict:
        """Get statistics including chunk counts."""
        base_stats = super().get_stats()
        base_stats.update(
            {
                "total_chunks": self.total_chunks,
                "completed_chunks": self.completed_chunks,
            }
        )
        return base_stats
