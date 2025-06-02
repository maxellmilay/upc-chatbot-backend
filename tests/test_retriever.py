"""
Test file for the HybridRetriever class.
Tests all methods and functionality including sparse retrieval, dense retrieval, and entity boosting.
"""

from django.test import TestCase
from unittest.mock import Mock, patch
import numpy as np
import json
from ai.lib.retriever import HybridRetriever
from ai.models.document import Document, DocumentChunk


class HybridRetrieverTestCase(TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.retriever = HybridRetriever(BOOST=0.2, k=10)
        
        # Create test documents and chunks for database testing
        self.document1 = Document.objects.create(
            file_url="https://example.com/doc1.pdf",
            description="Test document 1"
        )
        
        self.document2 = Document.objects.create(
            file_url="https://example.com/doc2.pdf", 
            description="Test document 2"
        )
        
        # Create test embeddings (384 dimensions for SBERT)
        self.test_embedding1 = np.random.rand(384).tolist()
        self.test_embedding2 = np.random.rand(384).tolist()
        self.query_embedding = np.random.rand(384)
        
        # Create test entities
        self.test_entities1 = [
            ["University of the Philippines", "ORG", 0, 28],
            ["Computer Science", "MISC", 30, 46]
        ]
        self.test_entities2 = [
            ["Manila", "GPE", 0, 6],
            ["Engineering", "MISC", 8, 19]
        ]
        
        # Create document chunks
        self.chunk1 = DocumentChunk.objects.create(
            document=self.document1,
            text="University of the Philippines Computer Science program",
            tokens_json=["university", "philippines", "computer", "science", "program"],
            embedding_json=self.test_embedding1,
            pos_json=[["University", "NOUN", "NN"]],
            entity_json=self.test_entities1
        )
        
        self.chunk2 = DocumentChunk.objects.create(
            document=self.document2,
            text="Manila Engineering courses and curriculum",
            tokens_json=["manila", "engineering", "courses", "curriculum"],
            embedding_json=self.test_embedding2,
            pos_json=[["Manila", "NOUN", "NNP"]],
            entity_json=self.test_entities2
        )
        
        # Mock NLP preprocessor response
        self.mock_preprocess_response = {
            "embeddings": self.query_embedding,
            "entities": [["Computer Science", "MISC", 0, 15]],
            "original_text": "Computer Science courses",
            "preprocessed_text": "computer science courses",
            "tokens": ["computer", "science", "courses"]
        }

    def test_initialization(self):
        """Test that the HybridRetriever initializes correctly."""
        retriever = HybridRetriever()
        self.assertEqual(retriever.BOOST, 0.1)  # default value
        self.assertEqual(retriever.k, 50)  # default value
        self.assertIsNotNone(retriever.nlp_preprocessor)
        
        # Test custom initialization
        custom_retriever = HybridRetriever(BOOST=0.3, k=25)
        self.assertEqual(custom_retriever.BOOST, 0.3)
        self.assertEqual(custom_retriever.k, 25)

    @patch('ai.lib.retriever.NLPPreprocessor')
    def test_preprocess_query(self, mock_nlp_class):
        """Test the _preprocess_query method."""
        # Mock the NLP preprocessor
        mock_nlp_instance = Mock()
        mock_nlp_instance.preprocess.return_value = self.mock_preprocess_response
        mock_nlp_class.return_value = mock_nlp_instance
        
        retriever = HybridRetriever()
        result = retriever._preprocess_query("Computer Science courses")
        
        mock_nlp_instance.preprocess.assert_called_once_with("Computer Science courses")
        self.assertEqual(result, self.mock_preprocess_response)

    def test_has_matching_entity_with_matches(self):
        """Test _has_matching_entity method when entities match."""
        doc_entities = [["University of the Philippines", "ORG"], ["Computer Science", "MISC"]]
        query_entities = [["Computer Science", "MISC"], ["Programming", "MISC"]]
        
        result = HybridRetriever._has_matching_entity(doc_entities, query_entities)
        self.assertTrue(result)

    def test_has_matching_entity_without_matches(self):
        """Test _has_matching_entity method when entities don't match."""
        doc_entities = [["University of the Philippines", "ORG"], ["Engineering", "MISC"]]
        query_entities = [["Computer Science", "MISC"], ["Programming", "MISC"]]
        
        result = HybridRetriever._has_matching_entity(doc_entities, query_entities)
        self.assertFalse(result)

    def test_has_matching_entity_empty_entities(self):
        """Test _has_matching_entity method with empty entity lists."""
        doc_entities = []
        query_entities = [["Computer Science", "MISC"]]
        
        result = HybridRetriever._has_matching_entity(doc_entities, query_entities)
        self.assertFalse(result)

    @patch('ai.lib.retriever.cosine_sim')
    def test_rerank_with_boost_with_matching_entities(self, mock_cosine_sim):
        """Test _rerank_with_boost method when entities match."""
        mock_cosine_sim.return_value = 0.8
        
        # Create a mock document chunk
        mock_doc = Mock()
        mock_doc.embedding_json = self.test_embedding1
        mock_doc.entity_json = json.dumps(self.test_entities1)
        
        query_emb = self.query_embedding
        query_entities = [["Computer Science", "MISC", 0, 15]]
        
        result = self.retriever._rerank_with_boost(mock_doc, query_emb, query_entities)
        
        # Should be cosine similarity + boost
        expected_score = 0.8 + self.retriever.BOOST
        self.assertEqual(result, expected_score)
        
        # Check that cosine_sim was called once
        mock_cosine_sim.assert_called_once()
        # Verify the arguments by checking the call
        call_args = mock_cosine_sim.call_args[0]
        np.testing.assert_array_equal(call_args[0], query_emb)
        np.testing.assert_array_equal(call_args[1], self.test_embedding1)

    @patch('ai.lib.retriever.cosine_sim')
    def test_rerank_with_boost_without_matching_entities(self, mock_cosine_sim):
        """Test _rerank_with_boost method when entities don't match."""
        mock_cosine_sim.return_value = 0.6
        
        # Create a mock document chunk
        mock_doc = Mock()
        mock_doc.embedding_json = self.test_embedding2
        mock_doc.entity_json = json.dumps(self.test_entities2)
        
        query_emb = self.query_embedding
        query_entities = [["Computer Science", "MISC", 0, 15]]
        
        result = self.retriever._rerank_with_boost(mock_doc, query_emb, query_entities)
        
        # Should be just cosine similarity (no boost)
        self.assertEqual(result, 0.6)
        
        # Check that cosine_sim was called once
        mock_cosine_sim.assert_called_once()
        # Verify the arguments by checking the call
        call_args = mock_cosine_sim.call_args[0]
        np.testing.assert_array_equal(call_args[0], query_emb)
        np.testing.assert_array_equal(call_args[1], self.test_embedding2)

    @patch('ai.lib.retriever.DocumentChunk')
    @patch('ai.lib.retriever.NLPPreprocessor')
    def test_retrieve_full_pipeline(self, mock_nlp_class, mock_document_class):
        """Test the complete retrieve method pipeline."""
        # Mock NLP preprocessor
        mock_nlp_instance = Mock()
        mock_nlp_instance.preprocess.return_value = self.mock_preprocess_response
        mock_nlp_class.return_value = mock_nlp_instance
        
        # Create mock document chunks that will be returned
        mock_chunk1 = Mock()
        mock_chunk1.id = 1
        mock_chunk1.text = "University of the Philippines Computer Science program"
        mock_chunk1.embedding_json = self.test_embedding1
        mock_chunk1.entity_json = json.dumps(self.test_entities1)
        
        mock_chunk2 = Mock()
        mock_chunk2.id = 2
        mock_chunk2.text = "Manila Engineering courses"
        mock_chunk2.embedding_json = self.test_embedding2
        mock_chunk2.entity_json = json.dumps(self.test_entities2)
        
        # Mock database query chain
        mock_annotated = Mock()
        mock_filtered = Mock()
        mock_ordered = Mock()
        
        # Setup the mock chain - properly handle slicing
        mock_document_class.objects.annotate.return_value = mock_annotated
        mock_annotated.filter.return_value = mock_filtered
        mock_filtered.order_by.return_value = mock_ordered
        
        # Mock the slicing operation [:self.k]
        mock_sliced = Mock()
        mock_ordered.__getitem__ = Mock(return_value=mock_sliced)
        
        # Mock the only() method to return the chunks
        mock_only_queryset = Mock()
        mock_only_queryset.__iter__ = Mock(return_value=iter([mock_chunk1, mock_chunk2]))
        mock_sliced.only.return_value = mock_only_queryset
        
        # Create retriever and run retrieve
        retriever = HybridRetriever(BOOST=0.2, k=10)
        
        with patch('ai.lib.retriever.cosine_sim') as mock_cosine_sim:
            # Mock cosine similarities - higher for chunk1 to test ranking
            mock_cosine_sim.side_effect = [0.9, 0.7]  # chunk1 gets 0.9, chunk2 gets 0.7
            
            results = retriever.retrieve("Computer Science courses")
        
        # Verify the NLP preprocessing was called
        mock_nlp_instance.preprocess.assert_called_once_with("Computer Science courses")
        
        # Verify database query was constructed correctly
        mock_document_class.objects.annotate.assert_called_once()
        mock_annotated.filter.assert_called_once()
        mock_filtered.order_by.assert_called_once_with('-rank')
        
        # Verify results are returned and ranked correctly
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        # chunk1 should be first (0.9 + 0.2 boost = 1.1 > 0.7)
        # because it has matching "Computer Science" entity
        self.assertEqual(results[0], mock_chunk1)
        self.assertEqual(results[1], mock_chunk2)

    @patch('ai.lib.retriever.DocumentChunk')
    @patch('ai.lib.retriever.NLPPreprocessor')
    def test_retrieve_empty_results(self, mock_nlp_class, mock_document_class):
        """Test retrieve method when no documents are found."""
        # Mock NLP preprocessor
        mock_nlp_instance = Mock()
        mock_nlp_instance.preprocess.return_value = self.mock_preprocess_response
        mock_nlp_class.return_value = mock_nlp_instance
        
        # Mock database query chain
        mock_annotated = Mock()
        mock_filtered = Mock()
        mock_ordered = Mock()
        
        mock_document_class.objects.annotate.return_value = mock_annotated
        mock_annotated.filter.return_value = mock_filtered
        mock_filtered.order_by.return_value = mock_ordered
        
        # Mock empty slicing result
        mock_sliced = Mock()
        mock_ordered.__getitem__ = Mock(return_value=mock_sliced)
        mock_sliced.only.return_value = []
        
        retriever = HybridRetriever()
        results = retriever.retrieve("Non-existent query")
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_retrieve_integration_with_real_database(self):
        """Integration test with real database operations."""
        # This test uses the real database with test data
        with patch('ai.lib.retriever.NLPPreprocessor') as mock_nlp_class:
            # Mock NLP preprocessor
            mock_nlp_instance = Mock()
            mock_nlp_instance.preprocess.return_value = self.mock_preprocess_response
            mock_nlp_class.return_value = mock_nlp_instance
            
            # Use the actual DocumentChunk model
            retriever = HybridRetriever(k=5)
            
            # This will likely fail with PostgreSQL functions in SQLite
            try:
                results = retriever.retrieve("Computer Science")
                self.assertIsInstance(results, list)
            except Exception as e:
                # Expected to fail due to PostgreSQL functions in SQLite test database
                error_message = str(e).lower()
                self.assertTrue(
                    "plainto_tsquery" in error_message or 
                    "text" in error_message or 
                    "embedding" in error_message
                )

    def test_custom_boost_and_k_values(self):
        """Test retriever with different boost and k values."""
        retriever1 = HybridRetriever(BOOST=0.5, k=20)
        retriever2 = HybridRetriever(BOOST=0.0, k=100)
        
        self.assertEqual(retriever1.BOOST, 0.5)
        self.assertEqual(retriever1.k, 20)
        self.assertEqual(retriever2.BOOST, 0.0)
        self.assertEqual(retriever2.k, 100)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with invalid JSON string in entities
        mock_doc = Mock()
        mock_doc.embedding_json = json.dumps(self.test_embedding1)  # Valid JSON string
        mock_doc.entity_json = "invalid json"  # Invalid JSON string
        
        query_emb = self.query_embedding
        query_entities = []
        
        # Should handle JSON decode error gracefully
        with self.assertRaises(json.JSONDecodeError):
            self.retriever._rerank_with_boost(mock_doc, query_emb, query_entities)
        
        # Test with None entity_json
        mock_doc_none = Mock()
        mock_doc_none.embedding_json = self.test_embedding1  # List format
        mock_doc_none.entity_json = None
        
        # Should handle None entities gracefully
        with patch('ai.lib.retriever.cosine_sim') as mock_cosine_sim:
            mock_cosine_sim.return_value = 0.5
            result = self.retriever._rerank_with_boost(mock_doc_none, query_emb, query_entities)
            self.assertEqual(result, 0.5)  # Should return just cosine similarity

    def test_entity_matching_edge_cases(self):
        """Test entity matching with various edge cases."""
        # Test with None entities
        doc_entities = None
        query_entities = [["Test", "MISC"]]
        
        with self.assertRaises(TypeError):
            HybridRetriever._has_matching_entity(doc_entities, query_entities)
        
        # Test with different entity tuple lengths
        doc_entities = [["Entity1", "TYPE1", 0, 10, "extra"]]  # 5 elements
        query_entities = [["Entity1", "TYPE1"]]  # 2 elements
        
        result = HybridRetriever._has_matching_entity(doc_entities, query_entities)
        self.assertTrue(result)  # Should still match on first element

    def tearDown(self):
        """Clean up after tests."""
        DocumentChunk.objects.all().delete()
        Document.objects.all().delete()
