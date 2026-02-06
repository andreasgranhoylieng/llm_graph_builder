"""
IngestService - Handles document loading and chunking with streaming support.
"""

import os
from typing import List, Generator, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.services.interfaces import IFileRepository


class IngestService:
    """Service for ingesting documents with streaming and batch support."""

    def __init__(self, file_repo: IFileRepository):
        self.file_repo = file_repo
        self._text_splitter = None

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy initialization of text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
            )
        return self._text_splitter

    def load_documents_from_folder(self, folder_path="./data") -> List:
        """
        Legacy method: Loads all PDF and TXT files from the data directory and chunks them.
        Use load_and_chunk_file for large-scale processing.
        """
        print(f"📂 Scanning '{folder_path}'...")

        pdf_files = self.file_repo.get_files(folder_path, [".pdf"])
        txt_files = self.file_repo.get_files(folder_path, [".txt"])

        documents = []

        for pdf in pdf_files:
            loader = PyPDFLoader(pdf)
            documents.extend(loader.load())

        for txt in txt_files:
            loader = TextLoader(txt)
            documents.extend(loader.load())

        return self._chunk_documents(documents)

    def load_and_chunk_file(
        self, file_path: str
    ) -> Tuple[List[Document], Optional[str]]:
        """
        Load and chunk a single file with error handling.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (chunks, error_message). error_message is None on success.
        """
        try:
            extension = os.path.splitext(file_path)[1].lower()

            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return [], f"Unsupported file type: {extension}"

            # Load document
            documents = loader.load()

            if not documents:
                return [], "No content extracted from file"

            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)
                doc.metadata["source_path"] = file_path

            # Chunk documents
            chunks = self.text_splitter.split_documents(documents)

            return chunks, None

        except Exception as e:
            return [], f"Error loading file: {str(e)}"

    def load_documents_streaming(
        self, file_paths: List[str]
    ) -> Generator[Tuple[str, List[Document], Optional[str]], None, None]:
        """
        Load documents one at a time to manage memory.

        Yields:
            Tuples of (file_path, chunks, error_message)
        """
        for file_path in file_paths:
            chunks, error = self.load_and_chunk_file(file_path)
            yield file_path, chunks, error

    def process_file_batch(
        self, file_paths: List[str]
    ) -> Tuple[List[Document], List[Tuple[str, str]]]:
        """
        Process a batch of files, returning all chunks and any errors.

        Args:
            file_paths: List of file paths to process

        Returns:
            Tuple of (all_chunks, errors) where errors is list of (filepath, error_message)
        """
        all_chunks = []
        errors = []

        for file_path in file_paths:
            chunks, error = self.load_and_chunk_file(file_path)

            if error:
                errors.append((file_path, error))
            else:
                all_chunks.extend(chunks)

        return all_chunks, errors

    def count_chunks_estimate(self, file_path: str) -> int:
        """
        Estimate the number of chunks a file will produce without loading it.
        Based on file size and average characters per page.
        """
        file_size = (
            self.file_repo.get_file_size(file_path)
            if hasattr(self.file_repo, "get_file_size")
            else 0
        )

        if file_size == 0:
            return 0

        # Rough estimate: 1 KB of PDF ≈ 500 characters of text
        # With 2000 char chunks, that's about 1 chunk per 4KB
        estimated_chunks = max(1, file_size // 4096)

        return estimated_chunks

    def _chunk_documents(self, documents, chunk_size=2000, chunk_overlap=200):
        """Splits documents into smaller chunks for the LLM."""
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"🔪 Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def get_folder_statistics(self, folder_path: str) -> dict:
        """Get statistics about files in a folder."""
        pdf_files = (
            self.file_repo.get_files_recursive(folder_path, [".pdf"])
            if hasattr(self.file_repo, "get_files_recursive")
            else self.file_repo.get_files(folder_path, [".pdf"])
        )
        txt_files = (
            self.file_repo.get_files_recursive(folder_path, [".txt"])
            if hasattr(self.file_repo, "get_files_recursive")
            else self.file_repo.get_files(folder_path, [".txt"])
        )

        all_files = pdf_files + txt_files

        total_size = 0
        if hasattr(self.file_repo, "get_file_size"):
            total_size = sum(self.file_repo.get_file_size(f) for f in all_files)

        estimated_chunks = sum(self.count_chunks_estimate(f) for f in all_files)

        return {
            "pdf_count": len(pdf_files),
            "txt_count": len(txt_files),
            "total_files": len(all_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "estimated_chunks": estimated_chunks,
            "files": all_files,
        }
