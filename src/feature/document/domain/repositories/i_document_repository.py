from abc import ABC, abstractmethod
from typing import Dict, Any

from src.feature.document.domain.entities.document import Document


class IDocumentRepository(ABC):
    """
    An interface for a document repository layer.
    This abstract class defines the methods for adding, retrieving, and searching documents.
    """
    
    @abstractmethod
    def save_document(self, 
                      user_id: str, 
                      file_id: str, 
                      node_id: str, 
                      document_embedding: list[float], 
                      details: Dict[str, Any]) -> str:
        """
        Adds a new document to the repository.
        
        :param document: A dictionary representing the document to add.
        """

    @abstractmethod
    def search_user_documents_by_embedding(self, user_id: str, embedding: list[float], offset: int = 0, limit: int = 10) -> list[Document]:

        """
        Searches for documents that match the given query.
        
        :param query: The search query string.
        :return: A list of dictionaries, each representing a document that matches the query.
        """

    @abstractmethod
    def update_user_document_by_id(self,
                                   user_id: str,
                                   file_id: str,
                                   node_id: str,
                                   document_embedding: list[float],
                                   details: Dict[str, Any]) -> str:
        """
        Updates a document in the repository.
        
        :param document_id: A string representing the ID of the document to update.
        """

    @abstractmethod
    def get_user_document_by_file_id(self, user_id: str, file_id: str) -> Dict[str, Any]:       
        """
        Retrieves a document from the repository by its fileId.
        
        :param fileId: The fileId of the document to retrieve.
        :return: A dictionary representing the retrieved document.
        """
