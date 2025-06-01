from rest_framework import serializers
from ai.models.conversation import Conversation, Message
from ai.serializers.document import DocumentChunkSerializer

class MessageSerializer(serializers.ModelSerializer):
    context = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = Message
        fields = ['id', 'conversation', 'role', 'content', 'context', 
                 'created_at', 'updated_at']

class ConversationSerializer(serializers.ModelSerializer):
    messages = serializers.SerializerMethodField()
    
    class Meta:
        model = Conversation
        fields = ['id', 'user', 'messages', 'created_at', 'updated_at']
    
    def get_messages(self, obj):
        return MessageSerializer(obj.messages, many=True).data
