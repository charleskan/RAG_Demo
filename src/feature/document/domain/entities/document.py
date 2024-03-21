from typing import Any, Dict, List

class Document:
    """
    Represents a document with user information, file information, document embedding, and details.
    """

    def __init__(self, 
                 document_id: str, 
                 user_id: str, 
                 file_id: str, 
                 document_embedding: List[float], 
                 details: Dict[str, Any]):
        self.document_id = document_id
        self.user_id = user_id
        self.file_id = file_id
        self.document_embedding = document_embedding
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the document.
        """
        return {
            "document_id": self.document_id,
            "user_id": self.user_id,
            "file_id": self.file_id,
            "document_embedding": self.document_embedding,
            "details": self.details
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Document':
        """
        Creates a new Document object from a dictionary.
        """
        return Document(
            document_id=data["document_id"],
            user_id=data["user_id"],
            file_id=data["file_id"],
            document_embedding=data["document_embedding"],
            details=data["details"]
        )