from abc import ABC, abstractmethod
from typing import Dict, Any

class IDocumentRepository(ABC):
    """
    An interface for a document repository layer.
    This abstract class defines the methods for adding, retrieving, and searching documents.
    """
    
    @abstractmethod
    def save_document(self,
            userId: str,
            fileId: str, 
            nodeId: str, 
            document_embedding: list[float], 
            details: Dict[str, Any]) -> str:
        """
        Adds a new document to the repository.
        
        :param document: A dictionary representing the document to add.
        """
        pass

    @abstractmethod
    def get_document_by_query_embedding(self,
                                        userId: str, 
                                        query_embedding: list[float],
                                        offset: int = 0,
                                        limit: int = 10,
                                        expression: str = None) -> list[dict[str, any]]:
        """
        Searches for documents that match the given query.
        
        :param query: The search query string.
        :return: A list of dictionaries, each representing a document that matches the query.
        """
        pass

    @abstractmethod
    def update_document_by_id(self,
                                  id: str, 
                                  userId: str, 
                                  fileId: str, 
                                  nodeId: str, 
                                  document_embedding: list[float], 
                                  details: Dict[str, Any]) -> str:
        """
        Updates a document in the repository.
        
        :param document: A dictionary representing the updated document.
        """
        pass

    @abstractmethod
    def get_document_by_fileId(self, userId: str, fileId: str) -> Dict[str, Any]:
        """
        Retrieves a document from the repository by its fileId.
        
        :param fileId: The fileId of the document to retrieve.
        :return: A dictionary representing the retrieved document.
        """
        pass

    @abstractmethod
    def get_document_by_fileId(self, userId: str, fileId: str) -> Dict[str, Any]:
        """
        Retrieves a document from the repository by its id.
        
        :param id: The id of the document to retrieve.
        :return: A dictionary representing the retrieved document.
        """
        pass