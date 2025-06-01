import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

class NLPPreprocessor:
    def __init__(self, text):
        self.text = text
        self.nlp_model = spacy.load("en_core_web_md")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process the text and extract features
        self.doc = self.nlp_model(self.text)
        self.tokens = self._extract_tokens()
        self.embeddings = self._extract_embeddings()
        self.pos = self._extract_pos()
        self.entities = self._extract_entities()
        
        self.preprocessed_text = self.preprocess()
    
    def get_data(self):
        """Return all processed data as a dictionary"""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "embeddings": self.embeddings,
            "pos": self.pos,
            "entities": self.entities,
            "preprocessed_text": self.preprocessed_text
        }
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        tokens = self._lemmatize()
        tokens = self._remove_stopwords(tokens)
        tokens = self._remove_punctuation(tokens)
        tokens = self._remove_numbers(tokens)
        return tokens
    
    def _extract_tokens(self):
        """Extract tokens from the text"""
        return [token.text for token in self.doc]
    
    def _lemmatize(self):
        """Lemmatize tokens"""
        return [token.lemma_.lower() for token in self.doc]
    
    def _remove_stopwords(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if not self.nlp_model.vocab[token].is_stop]
    
    def _remove_punctuation(self, tokens):
        """Remove punctuation from tokens"""
        return [token for token in tokens if not self.nlp_model.vocab[token].is_punct]
    
    def _remove_numbers(self, tokens):
        """Remove numeric tokens"""
        return [token for token in tokens if not token.isdigit()]
    
    def _extract_embeddings(self):
        """Extract sentence embeddings using SBERT"""
        return self.sbert_model.encode(self.text)
    
    def _extract_pos(self):
        """Extract part-of-speech tags"""
        return [(token.text, token.pos_, token.tag_) for token in self.doc]
    
    def _extract_entities(self):
        """Extract named entities"""
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in self.doc.ents]
