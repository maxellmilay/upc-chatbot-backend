import spacy
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast
import numpy as np
import os

class NLPPreprocessor:
    def __init__(self):
        self.nlp_model = spacy.load("en_core_web_md")
        
        # Setup ONNX model and tokenizer for all-MiniLM-L6-v2
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/all-MiniLM-L6-v2"))
        model_path = os.path.join(model_dir, "model_quantized.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        # Load tokenizer directly from tokenizer.json file
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Create ONNX Runtime session
        self.onnx_session = ort.InferenceSession(model_path)

    def get_data(self):
        """Return all processed data as a dictionary"""
        return {
            "original_text": self.original_text,
            "original_tokens": self.original_tokens,
            "embeddings": self.embeddings,
            "pos": self.pos,
            "entities": self.entities,
            "preprocessed_tokens": self.preprocessed_tokens,
            "preprocessed_text": self.preprocessed_text
        }
    
    def preprocess(self, original_text):
        """Main preprocessing pipeline"""
        self.original_text = original_text
        self.doc = self.nlp_model(original_text)

        self.original_tokens = self._extract_tokens()
        self.pos = self._extract_pos()
        self.entities = self._extract_entities()
        
        preprocessed_tokens = self._lemmatize()
        preprocessed_tokens = self._remove_stopwords(preprocessed_tokens)
        preprocessed_tokens = self._remove_punctuation(preprocessed_tokens)
        preprocessed_tokens = self._remove_whitespace_tokens(preprocessed_tokens)
        # preprocessed_tokens = self._remove_numbers(preprocessed_tokens)
        self.preprocessed_tokens = preprocessed_tokens
        self.preprocessed_text = " ".join(preprocessed_tokens)
        
        self.embeddings = self._extract_embeddings()

        return self.get_data()
    
    def _extract_tokens(self):
        """Extract tokens from the text"""
        return [token.text for token in self.doc]
    
    def _lemmatize(self):
        """Lemmatize tokens"""
        return [token.lemma_.lower() for token in self.doc]
    
    def _remove_stopwords(self, preprocessed_tokens):
        """Remove stopwords from tokens"""
        return [preprocessed_token for preprocessed_token in preprocessed_tokens if not self.nlp_model.vocab[preprocessed_token].is_stop]
    
    def _remove_punctuation(self, preprocessed_tokens):
        """Remove punctuation from tokens"""
        return [preprocessed_token for preprocessed_token in preprocessed_tokens if not self.nlp_model.vocab[preprocessed_token].is_punct]
    
    def _remove_whitespace_tokens(self, preprocessed_tokens):
        """Remove tokens that consist only of whitespace characters (spaces, newlines, tabs, etc.)"""
        return [preprocessed_token for preprocessed_token in preprocessed_tokens if preprocessed_token.strip()]
    
    # def _remove_numbers(self, preprocessed_tokens):
    #     """Remove numeric tokens"""
    #     return [preprocessed_token for preprocessed_token in preprocessed_tokens if not preprocessed_token.isdigit()]
    
    def _extract_embeddings(self):
        """Extract sentence embeddings using ONNX Runtime"""
        return self.get_embedding(self.preprocessed_text)
    
    def get_embedding(self, text: str):
        """Get embedding for a given text using the quantized ONNX model"""
        # Tokenize the input text
        encoded = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        
        # Run inference
        outputs = self.onnx_session.run(None, {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        })
        
        # The model has two outputs: [token_embeddings, sentence_embedding]
        # Use the pre-computed sentence embedding (index 1) for efficiency
        sentence_embedding = outputs[1]  # Shape: (batch_size, hidden_size)
        
        return sentence_embedding.squeeze() if sentence_embedding.shape[0] == 1 else sentence_embedding
    
    def _extract_pos(self):
        """Extract part-of-speech tags"""
        return [(token.text, token.pos_, token.tag_) for token in self.doc]
    
    def _extract_entities(self):
        """Extract named entities"""
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in self.doc.ents]
