import tempfile
import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from services.s3 import get_s3_service

class DocumentLoader:
    def __init__(self, s3_key: str):
        self.s3_key = s3_key
        self.file_extension = self._extract_file_extension(s3_key)
        self.s3_service = get_s3_service()
        self.pages = self.load(s3_key)
    
    def _extract_file_extension(self, s3_key: str) -> str:
        """Extract file extension from S3 key."""
        # Extract extension and remove the dot
        extension = os.path.splitext(s3_key)[1].lower().lstrip('.')
        return extension

    def load(self, s3_key: str) -> List[str]:
        """Load document based on file extension and return page content."""
        if self.file_extension == 'pdf':
            return self._load_pdf(s3_key)
        elif self.file_extension == 'docx':
            return self._load_docx(s3_key)
        else:
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
    
    def _load_pdf(self, s3_key: str) -> List[str]:
        """Load PDF document from S3 and return content per page."""
        # Download file content to memory
        file_content = self.s3_service.download_file_to_memory(s3_key)
        if file_content is None:
            raise ValueError(f"Failed to download file from S3: {s3_key}")
        
        # Create a temporary file to work with LangChain loaders
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Use LangChain PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            # Extract page content from each document
            pages = [doc.page_content.strip() for doc in documents if doc.page_content.strip()]
            
            return pages
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def _load_docx(self, s3_key: str) -> List[str]:
        """Load DOCX document from S3 and return content per logical section."""
        # Download file content to memory
        file_content = self.s3_service.download_file_to_memory(s3_key)
        if file_content is None:
            raise ValueError(f"Failed to download file from S3: {s3_key}")
        
        # Create a temporary file to work with LangChain loaders
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Use LangChain Docx2txtLoader
            loader = Docx2txtLoader(temp_file_path)
            documents = loader.load()
            
            # For DOCX, split content by page breaks or double newlines to simulate pages
            pages = []
            for doc in documents:
                content = doc.page_content.strip()
                if content:
                    # Split by page breaks (\f) or double newlines as logical page separators
                    page_splits = content.split('\f')  # Form feed character used for page breaks
                    if len(page_splits) == 1:
                        # If no page breaks found, split by double newlines as sections
                        page_splits = [section.strip() for section in content.split('\n\n') if section.strip()]
                    else:
                        page_splits = [page.strip() for page in page_splits if page.strip()]
                    
                    pages.extend(page_splits)
            
            # If no logical splits found, return the entire content as one page
            if not pages and documents:
                full_content = '\n'.join([doc.page_content.strip() for doc in documents if doc.page_content.strip()])
                if full_content:
                    pages = [full_content]
            
            return pages
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def get_pages(self) -> List[str]:
        """Get the loaded page contents as an array of strings."""
        return self.pages
