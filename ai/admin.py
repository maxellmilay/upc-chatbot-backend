from django.contrib import admin
from ai.models.document import Document, DocumentChunk
from ai.models.conversation import Conversation, Message

# Register your models here.
admin.site.register(Document)
admin.site.register(DocumentChunk)
admin.site.register(Conversation)
admin.site.register(Message)
