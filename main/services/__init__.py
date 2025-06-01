from main.lib.generic_api import GenericView
from ai.models.document import DocumentChunk, Document
from ai.serializers.document import DocumentSerializer
from ai.lib.loader import DocumentLoader
from django.db import transaction
from rest_framework.response import Response
from rest_framework import status

class DocumentView(GenericView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer

    def pre_create(self, request):
        pass

    def post_create(self, request, instance):
        loader = DocumentLoader(instance.file_url)
        pages = loader.get_pages()

        for page in pages:
            
            DocumentChunk.objects.create(
                document=instance,
                text=page
            )
    
    @transaction.atomic
    def create(self, request):
        if "create" not in self.allowed_methods:
            return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)

        self.pre_create(request)

        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            self.cache_object(serializer.data, instance.pk)
            self.invalidate_list_cache()

            self.post_create(request, instance)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)