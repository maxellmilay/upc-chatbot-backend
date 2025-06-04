from main.lib.generic_api import GenericView
from ai.models.conversation import Conversation
from ai.serializers.conversation import ConversationSerializer, SimpleConversationSerializer

class SimpleConversationView(GenericView):
    queryset = Conversation.objects.all()
    serializer_class = SimpleConversationSerializer
    allowed_methods = ['list']

class ConversationView(GenericView):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer
    allowed_methods = ['retrieve', 'create']
