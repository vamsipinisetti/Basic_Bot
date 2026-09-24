import unittest
from pathlib import Path

from src.config import load_config
from src.ingest import load_documents
from src.chunk import chunk_documents


class PipelineTests(unittest.TestCase):
    def test_config_loads_expected_keys(self):
        config = load_config()
        self.assertIn("GROQ_API_KEY", config)
        self.assertIn("GROQ_MODEL", config)
        self.assertIn("GROQ_BASE_URL", config)

    def test_document_ingestion_creates_sections(self):
        docs = load_documents(Path("data"))
        self.assertGreater(len(docs), 0)
        self.assertTrue(all("source" in doc and "page" in doc for doc in docs))

    def test_chunking_creates_overlapping_chunks(self):
        docs = load_documents(Path("data"))
        chunks = chunk_documents(docs, chunk_size=1000, overlap=200)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk["text"] for chunk in chunks))
        self.assertTrue(all("source" in chunk and "page" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
