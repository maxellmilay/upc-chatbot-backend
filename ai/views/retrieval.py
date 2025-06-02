from main.lib.generic_api import GenericView
from ai.lib.retriever import HybridRetriever
from ai.models.conversation import Message
from ai.serializers.conversation import MessageSerializer
from rest_framework.response import Response
from django.db import transaction
from rest_framework import status

class HybridRetrievalView(GenericView):
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    allowed_methods = ['create']
    
    @transaction.atomic
    def retrieve(self, request):
        self.pre_create(request)
        
        print(f'RETRIEVING SIMILARITY FOR: {request.data.get("query")}')

        retriever = HybridRetriever()
        result = retriever.retrieve(request.data.get('query'))
        return Response(result, status=status.HTTP_200_OK)
