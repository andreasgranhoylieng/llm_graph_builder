import os
from typing import List, Generator
from src.services.interfaces import IFileRepository


class FileRepository(IFileRepository):
    """Repository for file system operations with batch and recursive scanning support."""

    def get_files(self, folder_path: str, extensions: List[str]) -> List[str]:
        """Returns a list of file paths with specific extensions in the given folder."""
        if not os.path.exists(folder_path):
            return []

        files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                files.append(os.path.join(folder_path, file))
        return files

    def get_files_recursive(self, folder_path: str, extensions: List[str]) -> List[str]:
        """
        Recursively scan folders for files with specific extensions.
        Returns all matching files in the directory tree.
        """
        if not os.path.exists(folder_path):
            return []

        all_files = []

        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    all_files.append(os.path.join(root, file))

        return sorted(all_files)

    def get_files_batch(
        self,
        folder_path: str,
        extensions: List[str],
        skip_files: set = None,
        batch_size: int = 10,
    ) -> Generator[List[str], None, None]:
        """
        Yield batches of files, skipping already processed ones.

        Args:
            folder_path: Root directory to scan
            extensions: File extensions to include
            skip_files: Set of file paths to skip (already processed)
            batch_size: Number of files per batch

        Yields:
            Lists of file paths, batch_size at a time
        """
        skip_files = skip_files or set()

        # Get all files
        all_files = self.get_files_recursive(folder_path, extensions)

        # Filter out skipped files
        pending_files = [f for f in all_files if f not in skip_files]

        # Yield in batches
        for i in range(0, len(pending_files), batch_size):
            yield pending_files[i : i + batch_size]

    def count_files(self, folder_path: str, extensions: List[str]) -> int:
        """Count total files matching extensions in the directory tree."""
        return len(self.get_files_recursive(folder_path, extensions))

    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return 0

    def get_total_size(self, folder_path: str, extensions: List[str]) -> int:
        """Get total size of all matching files in bytes."""
        files = self.get_files_recursive(folder_path, extensions)
        return sum(self.get_file_size(f) for f in files)
