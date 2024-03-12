from abc import ABC, abstractmethod
from typing import Dict, Any

class IDocumentRepository(ABC):
    """
    An interface for a document repository layer.
    This abstract class defines the methods for adding, retrieving, and searching documents.
    """
    
    @abstractmethod
    def saveDocument(self, 
            nodeId: str, 
            document_embedding: list[float], 
            details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds a new document to the repository.
        
        :param document: A dictionary representing the document to add.
        """
        pass

    @abstractmethod
    def searchDocumentByQueryEmbedding(self, query_embedding: list[float]) -> Dict[str, Any]:
        """
        Searches for documents that match the given query.
        
        :param query: The search query string.
        :return: A list of dictionaries, each representing a document that matches the query.
        """
        pass