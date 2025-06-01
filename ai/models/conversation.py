from django.db import models
from django.contrib.auth.models import User

from ai.models.document import DocumentChunk

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Conversation for {self.user.username}"
    
    @property
    def messages(self):
        return Message.objects.filter(conversation=self)
    
class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    role = models.CharField(max_length=10)
    content = models.TextField()
    context = models.ManyToManyField(DocumentChunk, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Message for {self.conversation.user.username}"
