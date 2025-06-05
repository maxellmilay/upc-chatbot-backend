from main.lib.generic_api import GenericView
from ai.models.conversation import Conversation
from ai.serializers.conversation import ConversationSerializer, SimpleConversationSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.db.models import Q

class SimpleConversationView(GenericView):
    queryset = Conversation.objects.all()
    serializer_class = SimpleConversationSerializer
    allowed_methods = ['list']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def filter_queryset(self, filters, excludes):
        print(self.request)
        filter_q = Q(**filters)
        exclude_q = Q(**excludes)
        return self.queryset.filter(filter_q).exclude(exclude_q).filter(user=self.request.user)

class ConversationView(GenericView):
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer
    allowed_methods = ['retrieve', 'create']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def filter_queryset(self, filters, excludes):
        filter_q = Q(**filters)
        exclude_q = Q(**excludes)
        return self.queryset.filter(filter_q).exclude(exclude_q).filter(user=self.request.user)
