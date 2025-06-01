class DocumentLoader:
    def __init__(self, file_url):
        self.file_url = file_url
        self.file_extension = self._extract_file_extension(file_url)
        self.document = self._load(file_url)
        return self.document
    
    def _extract_file_extension(self):
        pass

    def _load(self, file_url):
        if self.file_extension == 'pdf':
            return self._load_pdf(self.file_url)
        elif self.file_extension == 'doc':
            return self._load_doc(self.file_url)
        elif self.file_extension == 'docx':
            return self._load_docx(self.file_url)
        else:
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
    
    def _load_pdf(self):
        pass
    
    def _load_doc(self):
        pass
    
    def _load_docx(self):
        pass
