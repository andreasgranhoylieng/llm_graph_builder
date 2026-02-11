"""
IngestService - Handles document loading and chunking with streaming support.
"""

import hashlib
import os
import re
from typing import Generator, List, Optional, Set, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
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
                chunk_size=config.INGEST_CHUNK_SIZE,
                chunk_overlap=config.INGEST_CHUNK_OVERLAP,
                length_function=len,
            )
        return self._text_splitter

    def load_documents_from_folder(self, folder_path="./data") -> List:
        """
        Legacy method: Loads all PDF and TXT files from the data directory and chunks them.
        Use load_and_chunk_file for large-scale processing.
        """
        print(f"Scanning '{folder_path}'...")

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
                documents = (
                    list(loader.lazy_load())
                    if callable(getattr(loader, "lazy_load", None))
                    else loader.load()
                )
            elif extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
            else:
                return [], f"Unsupported file type: {extension}"

            if not documents:
                return [], "No content extracted from file"

            for doc in documents:
                metadata = doc.metadata or {}
                metadata["source_file"] = os.path.basename(file_path)
                metadata["source_path"] = file_path
                doc.metadata = metadata

            chunks = self.text_splitter.split_documents(documents)
            chunks = self._optimize_chunks(chunks)
            self._assign_chunk_metadata(chunks, file_path=file_path)

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
        all_chunks: List[Document] = []
        errors: List[Tuple[str, str]] = []
        seen_fingerprints: Set[str] = set()

        for file_path in file_paths:
            chunks, error = self.load_and_chunk_file(file_path)

            if error:
                errors.append((file_path, error))
                continue

            for chunk in chunks:
                metadata = chunk.metadata or {}
                fingerprint = metadata.get("content_fingerprint")
                if not fingerprint:
                    normalized_text = self._normalize_text(chunk.page_content or "")
                    fingerprint = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()[
                        :16
                    ]
                    metadata["content_fingerprint"] = fingerprint
                    chunk.metadata = metadata

                if fingerprint in seen_fingerprints:
                    continue

                seen_fingerprints.add(fingerprint)
                all_chunks.append(chunk)

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

        # Rough estimate: 1 KB of PDF ~= 500 characters of text.
        # Scale from default 2000-char chunks ~= 4 KB per chunk.
        bytes_per_chunk = max(1024, int((config.INGEST_CHUNK_SIZE / 2000) * 4096))
        return max(1, file_size // bytes_per_chunk)

    def _chunk_documents(self, documents, chunk_size=None, chunk_overlap=None):
        """Splits documents into smaller chunks for the LLM."""
        if not documents:
            return []

        if chunk_size is None:
            chunk_size = config.INGEST_CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.INGEST_CHUNK_OVERLAP

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        chunks = self._optimize_chunks(chunks)
        self._assign_chunk_metadata(chunks)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def _normalize_text(self, text: str) -> str:
        """Normalize chunk text for dedupe and cleaner embeddings."""
        if not text:
            return ""
        if config.INGEST_NORMALIZE_WHITESPACE:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _optimize_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Improve chunk quality while reducing ingestion cost.

        - Normalize whitespace
        - Drop tiny/empty chunks
        - Deduplicate repeated chunks within a file
        """
        if not chunks:
            return []

        optimized: List[Document] = []
        seen_fingerprints: Set[str] = set()

        for chunk in chunks:
            normalized_text = self._normalize_text(chunk.page_content or "")
            if len(normalized_text) < config.INGEST_MIN_CHUNK_CHARS:
                continue

            fingerprint = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()[:16]

            if config.INGEST_DEDUP_WITHIN_FILE and fingerprint in seen_fingerprints:
                continue

            seen_fingerprints.add(fingerprint)
            chunk.page_content = normalized_text

            metadata = chunk.metadata or {}
            metadata["content_fingerprint"] = fingerprint
            metadata["char_count"] = len(normalized_text)
            chunk.metadata = metadata

            optimized.append(chunk)

        return optimized

    def filter_known_chunks(
        self, chunks: List[Document], seen_fingerprints: Set[str]
    ) -> Tuple[List[Document], int]:
        """
        Filter out chunks that were already seen in the current processing job.

        Returns:
            (unique_chunks, duplicate_count)
        """
        if not chunks:
            return [], 0

        unique_chunks: List[Document] = []
        duplicates = 0

        for chunk in chunks:
            metadata = chunk.metadata or {}
            fingerprint = metadata.get("content_fingerprint")
            if not fingerprint:
                normalized_text = self._normalize_text(chunk.page_content or "")
                fingerprint = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()[
                    :16
                ]
                metadata["content_fingerprint"] = fingerprint
                chunk.metadata = metadata

            if fingerprint in seen_fingerprints:
                duplicates += 1
                continue

            seen_fingerprints.add(fingerprint)
            unique_chunks.append(chunk)

        return unique_chunks, duplicates

    def _assign_chunk_metadata(
        self, chunks: List[Document], file_path: Optional[str] = None
    ) -> None:
        """Assign stable chunk identifiers so source retrieval can be precise."""
        if not chunks:
            return

        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            source_file = (
                metadata.get("source_file")
                or metadata.get("source")
                or (os.path.basename(file_path) if file_path else "unknown")
            )
            page = metadata.get("page", 0)

            normalized_content = self._normalize_text(chunk.page_content or "")
            content_hash = hashlib.md5(normalized_content.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{source_file}#p{page}:c{idx}:{content_hash}"

            metadata["chunk_id"] = chunk_id
            metadata["id"] = chunk_id
            metadata["content_fingerprint"] = metadata.get(
                "content_fingerprint", content_hash
            )
            metadata["char_count"] = metadata.get("char_count", len(normalized_content))
            chunk.metadata = metadata

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
