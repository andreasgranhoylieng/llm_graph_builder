from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.services.ingest_service import IngestService


def test_load_and_chunk_file_assigns_unique_chunk_ids():
    mock_file_repo = MagicMock()
    service = IngestService(mock_file_repo)

    chunk_a = Document(page_content="Chunk A", metadata={"page": 1})
    chunk_b = Document(page_content="Chunk B", metadata={"page": 1})
    splitter = MagicMock()
    splitter.split_documents.return_value = [chunk_a, chunk_b]
    service._text_splitter = splitter

    with patch("src.services.ingest_service.TextLoader") as loader_cls:
        loader = MagicMock()
        loader_cls.return_value = loader
        loader.load.return_value = [
            Document(page_content="Long document text", metadata={"source": "doc.txt"})
        ]

        chunks, error = service.load_and_chunk_file("C:/tmp/doc.txt")

    assert error is None
    assert len(chunks) == 2
    assert chunks[0].metadata["id"] != chunks[1].metadata["id"]
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["id"] == chunks[0].metadata["chunk_id"]


def test_chunk_documents_assigns_ids_for_legacy_path():
    mock_file_repo = MagicMock()
    service = IngestService(mock_file_repo)

    docs = [Document(page_content="Alpha", metadata={"source": "legacy.txt"})]
    chunks = service._chunk_documents(docs, chunk_size=50, chunk_overlap=0)

    assert chunks
    assert "id" in chunks[0].metadata
    assert "chunk_id" in chunks[0].metadata


def test_optimize_chunks_deduplicates_by_normalized_content():
    mock_file_repo = MagicMock()
    service = IngestService(mock_file_repo)

    chunk_a = Document(page_content="GPT-4   is   powerful.", metadata={})
    chunk_b = Document(page_content="GPT-4 is powerful.", metadata={})
    chunk_c = Document(page_content="Claude is safe.", metadata={})

    optimized = service._optimize_chunks([chunk_a, chunk_b, chunk_c])

    assert len(optimized) == 2
    assert all("content_fingerprint" in c.metadata for c in optimized)
    assert all("char_count" in c.metadata for c in optimized)


def test_filter_known_chunks_skips_already_seen_fingerprints():
    mock_file_repo = MagicMock()
    service = IngestService(mock_file_repo)

    first = Document(page_content="Repeated text", metadata={})
    second = Document(page_content="Repeated text", metadata={})
    unique = Document(page_content="Different text", metadata={})

    optimized = service._optimize_chunks([first, second, unique])
    seen = {optimized[0].metadata["content_fingerprint"]}

    remaining, duplicate_count = service.filter_known_chunks(optimized, seen)

    assert duplicate_count == 1
    assert len(remaining) == 1
    assert remaining[0].page_content == "Different text"
