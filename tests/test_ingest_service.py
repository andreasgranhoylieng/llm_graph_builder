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
