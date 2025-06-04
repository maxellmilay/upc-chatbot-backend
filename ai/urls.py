from django.urls import path
from ai.views.loader import DocumentView, DocumentChunkView, SimpleDocumentChunkView
from ai.views.conversation import ConversationView, SimpleConversationView
from ai.views.retrieval import HybridRetrievalView

urlpatterns = [
    # Document
    path('document/', DocumentView.as_view({'post': 'create', 'get': 'list'}), name='document'),
    path('document/<int:pk>/', DocumentView.as_view({'get': 'retrieve', 'delete': 'destroy'}), name='document-detail'),
    
    # Document Chunk
    path('simple-document-chunk/', SimpleDocumentChunkView.as_view({'get': 'list'}), name='simple-document-chunk'),
    path('document-chunk/<int:pk>/', DocumentChunkView.as_view({'get': 'retrieve'}), name='document-chunk-detail'),

    # Retrieval
    path('retrieve/', HybridRetrievalView.as_view({'post': 'create'}), name='retrieval'),
    
    # Conversation
    path('simple-conversation/', SimpleConversationView.as_view({'get': 'list'}), name='simple-conversation'),
    path('conversation/<int:pk>/', ConversationView.as_view({'get': 'retrieve'}), name='conversation-detail'),
    path('conversation/', ConversationView.as_view({'post': 'create'}), name='conversation'),
]
