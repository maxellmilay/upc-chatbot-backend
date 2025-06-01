from rest_framework import serializers
from ai.models.document import Document, DocumentChunk

class DocumentSerializer(serializers.ModelSerializer):
    chunks = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = ['id', 'file_url', 'created_at', 'updated_at', 'chunks']
    
    def get_chunks(self, obj):
        return DocumentChunkSerializer(obj.chunks, many=True).data
        
class DocumentChunkSerializer(serializers.ModelSerializer):
    tokens = serializers.SerializerMethodField()
    embeddings = serializers.SerializerMethodField()
    pos = serializers.SerializerMethodField()
    entities = serializers.SerializerMethodField()
    
    class Meta:
        model = DocumentChunk
        fields = ['id', 'document', 'text', 'tokens', 'embeddings', 'pos', 'entities', 
                 'created_at', 'updated_at']
    
    def get_tokens(self, obj):
        return obj.tokens
    
    def get_embeddings(self, obj):
        return obj.embeddings
    
    def get_pos(self, obj):
        return obj.pos
    
    def get_entities(self, obj):
        return obj.entities
