import spacy

class NLPPreprocessor:
    def __init__(self, text):
        self.text = text
        self.nlp_model = spacy.load("en_core_web_sm")
        self.tokens = self._extract_tokens()
        self.embeddings = self._extract_embeddings()
        self.pos = self._extract_pos()
        self.entities = self._extract_entities()
        
        self.preprocessed_text = self.preprocess()
        
        data = {
            "text": self.text,
            "tokens": self.tokens,
            "embeddings": self.embeddings,
            "pos": self.pos,
            "entities": self.entities,
            "preprocessed_text": self.preprocessed_text
        }

        return data
    
    def preprocess(self):
        tokens = self._lemmatize(self.text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._remove_punctuation(tokens)
        tokens = self._remove_numbers(tokens)

        return tokens
    
    def _lemmatize(self):
        pass
    
    def _remove_stopwords(self):
        pass
    
    def _remove_punctuation(self):
        pass
    
    def _remove_numbers(self):
        pass
    
    def _extract_embeddings(self):
        pass
    
    def _extract_pos(self):
        pass
    
    def _extract_entities(self):
        pass
