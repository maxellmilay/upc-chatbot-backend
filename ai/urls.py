from django.urls import path
from ai.views.loader import DocumentView, DocumentChunkView, SimpleDocumentChunkView

urlpatterns = [
    # Document
    path('document/', DocumentView.as_view({'post': 'create', 'get': 'list'}), name='document'),
    path('document/<int:pk>/', DocumentView.as_view({'get': 'retrieve', 'delete': 'destroy'}), name='document-detail'),
    
    # Document Chunk
    path('simple-document-chunk/', SimpleDocumentChunkView.as_view({'get': 'list'}), name='document-chunk'),
    path('document-chunk/<int:pk>/', DocumentChunkView.as_view({'get': 'retrieve'}), name='document-chunk-detail'),
]
