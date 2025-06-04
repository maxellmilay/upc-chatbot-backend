from rest_framework import serializers
from ai.models.conversation import Conversation, Message
from ai.serializers.document import ContextDocumentChunkSerializer

class MessageSerializer(serializers.ModelSerializer):
    context = ContextDocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'context', 'created_at', 'updated_at']

class ConversationSerializer(serializers.ModelSerializer):
    messages = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = Conversation
        fields = ['id', 'user', 'title', 'messages', 'created_at', 'updated_at']
    
    def get_messages(self, obj):
        return MessageSerializer(obj.messages, many=True).data

class SimpleConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'updated_at']
