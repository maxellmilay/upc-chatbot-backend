from ai.lib.nlp import NLPPreprocessor
from ai.models.document import Document
from django.contrib.postgres.search import SearchRank, SearchVector
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from ai.utils.retrieval import cosine_sim

class HybridRetriever():
    def __init__(self, query, BOOST=0.1, k=50):
        self.query = query
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        self.BOOST = BOOST
        self.k = k
        
        result = self._retrieve(query)
        
        return result
        
    def _preprocess_query(self):
        return NLPPreprocessor(self.query)
    
    def _has_matching_entity(doc_entities, query_entities):
        doc_ents = set([ent[0] for ent in doc_entities])
        query_ents = set([ent[0] for ent in query_entities])
        return not doc_ents.isdisjoint(query_ents)
    
    def _retrieve(self, query):
        preprocessed_query = self._preprocess_query()
        
        query_emb = preprocessed_query.get("embeddings")
        query_entities = preprocessed_query.get("entities")
        
        # === Sparse retrieval using PostgreSQL full-text search ===
        docs = Document.objects.annotate(
            rank=SearchRank(SearchVector('text'), query)
        ).filter(rank__gte=0.1).order_by('-rank')[:self.k]
        
        candidate_docs = docs.only("id", "text", "embedding_json", "entities_json")
        
        # === Dense embedding + entity-boosted reranking ===
        reranked = sorted(
            candidate_docs,
            key=lambda doc: self._rerank_with_boost(doc, query_emb, query_entities),
            reverse=True
        )
        
        return reranked

    def _rerank_with_boost(self, doc, query_emb, query_entities):
        emb = np.array(json.loads(doc.embedding_json))
        ents = json.loads(doc.entities_json)
        sim = cosine_sim(query_emb, emb)
        if self._has_matching_entity(ents, query_entities):
            sim += self.BOOST
        return sim
