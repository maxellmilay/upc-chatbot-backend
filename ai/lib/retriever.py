from ai.lib.nlp import NLPPreprocessor
from ai.models.document import DocumentChunk
from django.contrib.postgres.search import SearchRank, SearchVector
from ai.utils.retrieval import cosine_sim
import numpy as np
import json

class HybridRetriever():
    def __init__(self, BOOST=0.1, sparse_k=10, dense_k=3):
        self.nlp_preprocessor = NLPPreprocessor()
        self.BOOST = BOOST
        self.sparse_k = sparse_k
        self.dense_k = dense_k
        
    def _preprocess_query(self, query):
        return self.nlp_preprocessor.preprocess(query)
    
    @staticmethod
    def _has_matching_entity(doc_entities, query_entities):
        doc_ents = set([ent[0] for ent in doc_entities])
        query_ents = set([ent[0] for ent in query_entities])
        return not doc_ents.isdisjoint(query_ents)
    
    def retrieve(self, query):
        preprocessed_query = self._preprocess_query(query)
        
        query_emb = preprocessed_query.get("embeddings")
        query_entities = preprocessed_query.get("entities")
        
        print("ATTEMPTING SPARSE RETRIEVAL")
        
        # === Sparse retrieval using PostgreSQL full-text search ===
        docs = DocumentChunk.objects.annotate(
            rank=SearchRank(SearchVector('text'), query)
        ).order_by('-rank')[:self.sparse_k]
        
        candidate_docs = docs.only("id", "text", "embedding_json", "entity_json")
        
        print(f'SPARSE RETRIEVAL RESULTS: {candidate_docs}')
        
        print("ATTEMPTING DENSE RETRIEVAL")
        
        # === Dense embedding + entity-boosted reranking ===
        reranked = sorted(
            candidate_docs,
            key=lambda doc: self._rerank_with_boost(doc, query_emb, query_entities),
            reverse=True
        )[:self.dense_k]
        
        print(f'DENSE RETRIEVAL RESULTS: {reranked}')
        
        # Convert DocumentChunk objects to dictionaries for JSON serialization
        result = []
        for doc in reranked:
            result.append({
                'id': doc.id,
                'text': doc.text,
            })

        return result

    def _rerank_with_boost(self, doc, query_emb, query_entities):
        # Handle both JSON string and list formats for embeddings
        if isinstance(doc.embedding_json, str):
            emb = np.array(json.loads(doc.embedding_json))
        else:
            emb = np.array(doc.embedding_json)
        
        # Handle both JSON string and list formats for entities
        if isinstance(doc.entity_json, str):
            ents = json.loads(doc.entity_json) if doc.entity_json else []
        else:
            ents = doc.entity_json if doc.entity_json else []
            
        sim = cosine_sim(query_emb, emb)
        if self._has_matching_entity(ents, query_entities):
            sim += self.BOOST
        return sim
