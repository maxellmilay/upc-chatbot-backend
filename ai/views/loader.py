from main.lib.generic_api import GenericView
from ai.models.document import Document, DocumentChunk
from ai.serializers.document import DocumentSerializer, DocumentChunkSerializer, SimpleDocumentChunkSerializer
from ai.lib.loader import DocumentLoader
from ai.lib.nlp import NLPPreprocessor
from rest_framework.permissions import IsAdminUser
from rest_framework_simplejwt.authentication import JWTAuthentication

class DocumentView(GenericView):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    allowed_methods = ['create', 'list', 'retrieve', 'destroy']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAdminUser]
    
    def post_create(self, request, instance):
        print("\nCREATING DOCUMENT CHUNKS\n")
        print(f"Loading document from {instance.file_url}")
        loader = DocumentLoader(instance.file_url)
        print(f"Loaded {len(loader.pages)} pages")
        nlp_processor = NLPPreprocessor()
        pages = loader.get_pages()
        page_count = 0
        total_pages = len(pages)
        
        for page in pages:
            page_count += 1
            print(f"Processing page {page_count}/{total_pages}")

            nlp_data = nlp_processor.preprocess(page)

            DocumentChunk.objects.create(
                document=instance,
                text=nlp_data["original_text"],
                tokens_json=nlp_data["preprocessed_tokens"],
                embedding_json=nlp_data["embeddings"].tolist(),
                pos_json=nlp_data["pos"],
                entity_json=nlp_data["entities"]
            )

class DocumentChunkView(GenericView):
    queryset = DocumentChunk.objects.all()
    serializer_class = DocumentChunkSerializer
    allowed_methods = ['retrieve']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAdminUser]
        
class SimpleDocumentChunkView(GenericView):
    queryset = DocumentChunk.objects.all()
    serializer_class = SimpleDocumentChunkSerializer
    allowed_methods = ['list']
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAdminUser]
