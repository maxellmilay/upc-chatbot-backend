from django.test import TestCase
from ai.lib.loader import DocumentLoader

class DocumentLoaderTestCase(TestCase):
    def test_document_loader(self):
        loader = DocumentLoader("knowledge-base/Student-Handbook-2022.pdf")
        self.assertIsNotNone(loader.pages)
        self.assertGreater(len(loader.pages), 0)
        
        pages = loader.get_pages()
        for i, page in enumerate(pages):
            print(f"Page {i + 1} (first 100 chars): {page[:100]}")
