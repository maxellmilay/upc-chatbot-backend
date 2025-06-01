from django.db import models
import json

class Document(models.Model):
    file_url = models.URLField(max_length=2000)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.file_url
    
    @property
    def chunks(self):
        return DocumentChunk.objects.filter(document=self)
    
class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    text = models.TextField()
    tokens_json = models.JSONField()
    embedding_json = models.JSONField()
    pos_json = models.JSONField()
    entity_json = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chunk for {self.document.file_url}"
    
    @property
    def tokens(self):
        return json.loads(self.tokens_json)
    
    @property
    def embeddings(self):
        return json.loads(self.embedding_json)
    
    @property
    def pos(self):
        return json.loads(self.pos_json)
    
    @property
    def entities(self):
        return json.loads(self.entity_json)
