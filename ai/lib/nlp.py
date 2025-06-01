import spacy
from sentence_transformers import SentenceTransformer

class NLPPreprocessor:
    def __init__(self, text):
        self.original_text = text
        self.nlp_model = spacy.load("en_core_web_md")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process the text and extract features
        self.doc = self.nlp_model(self.original_text)

        self.original_tokens = self._extract_tokens()
        self.pos = self._extract_pos()
        self.entities = self._extract_entities()
        
        self.preprocessed_text, self.preprocessed_tokens = self.preprocess()
        
        self.embeddings = self._extract_embeddings()
    
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
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        preprocessed_tokens = self._lemmatize()
        preprocessed_tokens = self._remove_stopwords(preprocessed_tokens)
        preprocessed_tokens = self._remove_punctuation(preprocessed_tokens)
        # preprocessed_tokens = self._remove_numbers(preprocessed_tokens)
        processed_text = " ".join(preprocessed_tokens)

        return processed_text, preprocessed_tokens
    
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
    
    # def _remove_numbers(self, preprocessed_tokens):
    #     """Remove numeric tokens"""
    #     return [preprocessed_token for preprocessed_token in preprocessed_tokens if not preprocessed_token.isdigit()]
    
    def _extract_embeddings(self):
        """Extract sentence embeddings using SBERT"""
        return self.sbert_model.encode(self.preprocessed_text)
    
    def _extract_pos(self):
        """Extract part-of-speech tags"""
        return [(token.text, token.pos_, token.tag_) for token in self.doc]
    
    def _extract_entities(self):
        """Extract named entities"""
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in self.doc.ents]
