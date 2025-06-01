from django.test import TestCase
import numpy as np
from ai.lib.nlp import NLPPreprocessor


class NLPPreprocessorTestCase(TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_text = "Hello world! My name is John Smith and I live in New York City. I have 25 apples. The quick brown fox jumps over the lazy dog."
        self.nlp_processor = NLPPreprocessor(self.test_text)
    
    def test_initialization(self):
        """Test that the NLPPreprocessor initializes correctly."""
        self.assertEqual(self.nlp_processor.original_text, self.test_text)
        self.assertIsNotNone(self.nlp_processor.nlp_model)
        self.assertIsNotNone(self.nlp_processor.sbert_model)
        self.assertIsNotNone(self.nlp_processor.doc)
        self.assertIsNotNone(self.nlp_processor.original_tokens)
        self.assertIsNotNone(self.nlp_processor.embeddings)
        self.assertIsNotNone(self.nlp_processor.pos)
        self.assertIsNotNone(self.nlp_processor.entities)
        self.assertIsNotNone(self.nlp_processor.preprocessed_text)
        self.assertIsNotNone(self.nlp_processor.preprocessed_tokens)
    
    def test_get_data(self):
        """Test that get_data returns the expected dictionary structure."""
        data = self.nlp_processor.get_data()
        
        self.assertIsInstance(data, dict)
        self.assertIn("original_text", data)
        self.assertIn("original_tokens", data)
        self.assertIn("embeddings", data)
        self.assertIn("pos", data)
        self.assertIn("entities", data)
        self.assertIn("preprocessed_text", data)
        self.assertIn("preprocessed_tokens", data)
        
        self.assertEqual(data["original_text"], self.test_text)
        self.assertIsInstance(data["original_tokens"], list)
        self.assertIsInstance(data["embeddings"], np.ndarray)
        self.assertIsInstance(data["pos"], list)
        self.assertIsInstance(data["entities"], list)
        self.assertIsInstance(data["preprocessed_text"], str)
        self.assertIsInstance(data["preprocessed_tokens"], list)
        
        print(f"\n\nBEFORE NLP PROCESSING: {len(data['original_tokens'])}")
        print(f"BEFORE TEXT: {data['original_text']}")
        print(f"AFTER NLP PROCESSING: {len(data['preprocessed_tokens'])}")
        print(f"AFTER TEXT: {data['preprocessed_text']}\n")
    
    def test_extract_tokens(self):
        """Test token extraction functionality."""
        tokens = self.nlp_processor.original_tokens
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)
    
    def test_extract_embeddings(self):
        """Test sentence embedding extraction."""
        embeddings = self.nlp_processor.embeddings
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertGreater(len(embeddings), 0)
        # SBERT typically produces 384-dimensional embeddings for all-MiniLM-L6-v2
        self.assertEqual(embeddings.shape[0], 384)
    
    def test_extract_pos(self):
        """Test part-of-speech tag extraction."""
        pos_tags = self.nlp_processor.pos
        
        self.assertIsInstance(pos_tags, list)
        self.assertGreater(len(pos_tags), 0)
        
        # Each POS tag should be a tuple with (text, pos, tag)
        for pos_tag in pos_tags:
            self.assertIsInstance(pos_tag, tuple)
            self.assertEqual(len(pos_tag), 3)
            self.assertIsInstance(pos_tag[0], str)  # text
            self.assertIsInstance(pos_tag[1], str)  # pos
            self.assertIsInstance(pos_tag[2], str)  # tag
    
    def test_extract_entities(self):
        """Test named entity extraction."""
        entities = self.nlp_processor.entities
        
        self.assertIsInstance(entities, list)
        
        # Check if John Smith is recognized as a person
        entity_texts = [entity[0] for entity in entities]
        self.assertIn("John Smith", entity_texts)
        
        # Check if New York City is recognized as a location
        # Note: This might vary depending on spaCy's recognition
        location_entities = [entity for entity in entities if entity[1] in ["GPE", "LOC"]]
        self.assertGreater(len(location_entities), 0)
        
        # Each entity should be a tuple with (text, label, start, end)
        for entity in entities:
            self.assertIsInstance(entity, tuple)
            self.assertEqual(len(entity), 4)
            self.assertIsInstance(entity[0], str)  # text
            self.assertIsInstance(entity[1], str)  # label
            self.assertIsInstance(entity[2], int)  # start_char
            self.assertIsInstance(entity[3], int)  # end_char
    
    def test_lemmatize(self):
        """Test lemmatization functionality."""
        # Create a simple processor to test lemmatization
        test_processor = NLPPreprocessor("running dogs are barking")
        lemmatized = test_processor._lemmatize()
        
        self.assertIsInstance(lemmatized, list)
        self.assertIn("run", lemmatized)  # "running" should be lemmatized to "run"
        self.assertIn("dog", lemmatized)  # "dogs" should be lemmatized to "dog"
        self.assertIn("bark", lemmatized)  # "barking" should be lemmatized to "bark"
    
    def test_remove_stopwords(self):
        """Test stopword removal functionality."""
        tokens = ["the", "quick", "brown", "fox", "is", "jumping"]
        filtered_tokens = self.nlp_processor._remove_stopwords(tokens)
        
        self.assertIsInstance(filtered_tokens, list)
        self.assertNotIn("the", filtered_tokens)
        self.assertNotIn("is", filtered_tokens)
        self.assertIn("quick", filtered_tokens)
        self.assertIn("brown", filtered_tokens)
        self.assertIn("fox", filtered_tokens)
        self.assertIn("jumping", filtered_tokens)
    
    def test_remove_punctuation(self):
        """Test punctuation removal functionality."""
        tokens = ["hello", "!", "world", ".", "how", "?"]
        filtered_tokens = self.nlp_processor._remove_punctuation(tokens)
        
        self.assertIsInstance(filtered_tokens, list)
        self.assertNotIn("!", filtered_tokens)
        self.assertNotIn(".", filtered_tokens)
        self.assertNotIn("?", filtered_tokens)
        self.assertIn("hello", filtered_tokens)
        self.assertIn("world", filtered_tokens)
        self.assertIn("how", filtered_tokens)
    
    def test_remove_numbers(self):
        """Test number removal functionality."""
        tokens = ["hello", "123", "world", "456", "test"]
        filtered_tokens = self.nlp_processor._remove_numbers(tokens)
        
        self.assertIsInstance(filtered_tokens, list)
        self.assertNotIn("123", filtered_tokens)
        self.assertNotIn("456", filtered_tokens)
        self.assertIn("hello", filtered_tokens)
        self.assertIn("world", filtered_tokens)
        self.assertIn("test", filtered_tokens)
    
    def test_preprocess_pipeline(self):
        """Test the complete preprocessing pipeline."""
        preprocessed_tokens = self.nlp_processor.preprocessed_tokens
        preprocessed_text = self.nlp_processor.preprocessed_text
        
        self.assertIsInstance(preprocessed_tokens, list)
        self.assertIsInstance(preprocessed_text, str)
        self.assertGreater(len(preprocessed_tokens), 0)
        
        # Check that preprocessing removed stopwords, punctuation, and numbers
        # The exact results depend on spaCy's processing
        for token in preprocessed_tokens:
            self.assertIsInstance(token, str)
            self.assertNotEqual(token, "!")
            self.assertNotEqual(token, ".")
            self.assertFalse(token.isdigit())
    
    def test_different_text_inputs(self):
        """Test the processor with different types of text input."""
        # Test with empty string
        empty_processor = NLPPreprocessor("")
        self.assertEqual(empty_processor.original_text, "")
        self.assertIsInstance(empty_processor.original_tokens, list)
        
        # Test with only punctuation
        punct_processor = NLPPreprocessor("!@#$%^&*()")
        self.assertIsInstance(punct_processor.original_tokens, list)
        
        # Test with only numbers
        num_processor = NLPPreprocessor("123 456 789")
        self.assertIsInstance(num_processor.original_tokens, list)
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent for the same text."""
        processor1 = NLPPreprocessor("This is a test sentence.")
        processor2 = NLPPreprocessor("This is a test sentence.")
        
        # Embeddings should be identical for the same text
        np.testing.assert_array_almost_equal(
            processor1.embeddings, 
            processor2.embeddings, 
            decimal=6
        )
    
    def test_nlp_processor_output_format(self):
        """Test that NLP processor outputs are in the expected format for chatbot integration."""
        # Test with a typical chatbot query
        chatbot_query = "What are the admission requirements for Computer Science?"
        processor = NLPPreprocessor(chatbot_query)
        
        # Verify the output structure
        data = processor.get_data()
        self.assertIsInstance(data, dict)
        
        # Assertions for chatbot-specific requirements
        self.assertGreater(len(data['preprocessed_tokens']), 0)
        self.assertEqual(len(data['embeddings']), 384)  # SBERT embedding dimension 
