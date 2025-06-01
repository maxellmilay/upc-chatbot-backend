from django.urls import path
from ai.views.loader import DocumentView, SimpleDocumentView

urlpatterns = [
    path('simple-document/', SimpleDocumentView.as_view({'get': 'list'}), name='simple-document'),
    path('document', DocumentView.as_view({'post': 'create'}), name='document'),
    path('document/<int:pk>', DocumentView.as_view({'get': 'retrieve', 'delete': 'destroy'}), name='document-detail'),
]
