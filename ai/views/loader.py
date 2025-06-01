from main.lib.generic_api import GenericView
from ai.models.document import Document, DocumentChunk
from ai.serializers.document import DocumentSerializer, SimpleDocumentSerializer
from ai.lib.loader import DocumentLoader
from ai.lib.nlp import NLPPreprocessor

class DocumentView(GenericView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    allowed_methods = ['create', 'list', 'retrieve', 'destroy']

    def pre_create(self, request):
        pass
    
    def post_create(self, request, instance):
        loader = DocumentLoader()
        nlp_processor = NLPPreprocessor()
        pages = loader.get_pages()
        
        for page in pages:
            nlp_data = nlp_processor.preprocess(page)

            DocumentChunk.objects.create(
                document=instance,
                text=nlp_data["preprocessed_text"],
                tokens_json=nlp_data["preprocessed_tokens"],
                embedding_json=nlp_data["embeddings"],
                pos_json=nlp_data["pos"],
                entity_json=nlp_data["entities"]
            )

class SimpleDocumentView(GenericView):
    queryset = Document.objects.all()
    serializer_class = SimpleDocumentSerializer
    allowed_methods = ['list']
