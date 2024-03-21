from typing import Any, Dict
from pymilvus import connections, Collection
import os


from src.feature.document.domain.repositories.i_document_repository import IDocumentRepository

class MilvusDocumentRepository(IDocumentRepository):
    """
    A repository class for interacting with a Milvus collection to save, retrieve, and update documents.
    """

    def __init__(self):
        """
        Initializes the MilvusDocumentRepository class.

        Connects to the Milvus server using the provided host and port environment variables.
        Creates a collection named "book03" to store the documents.
        """
        connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))
        self.collection = Collection("book03")

    def save_document(self,
                      user_id: str,
                      file_id: str,
                      node_id: str,
                      document_embedding: list[float],
                      details: Dict[str, Any]) -> str:
        """
        Saves a document to the Milvus collection.

        Args:
            userId (str): The ID of the user who owns the document.
            fileId (str): The ID of the file associated with the document.
            nodeId (str): The ID of the text node associated with the document.
            document_embedding (list[float]): The embedding vector representation of the document.
            details (Dict[str, Any]): Additional details or metadata of the document.

        Returns:
            str: The primary key of the saved document.

        Raises:
            Exception: If an error occurs during the document save operation.
        """
        try:
            self.collection.load()

            document = {
                "userId": user_id,
                "fileId": file_id,
                "textNodeId": node_id,
                "embedding": document_embedding,
                "details": details
            }
            
            result = self.collection.insert(data=document)

            index_params = {
                "index_type": "IVF_FLAT",  # index type
                "metric_type": "L2",  # distance metric
                "params": {"nlist": 1024}  # index params
            }

            self.collection.create_index(field_name="embedding", index_params=index_params)

            self.collection.release()

            return result.primary_keys
        
        except Exception as e:
            print(f"An error occurred: {e}")

            return {"error": "An unexpected error occurred during the document save operation."}

    
    def get_user_documents_by_query_embedding(self,
                                            user_id: str, 
                                            query_embedding: list[float],
                                            offset: int = 0,
                                            limit: int = 10) -> list[dict[str, any]]:
        """
        Retrieves documents from the Milvus collection based on a query embedding vector.

        Args:
            userId (str): The ID of the user who owns the documents.
            query_embedding (list[float]): The embedding vector representation of the query.
            offset (int, optional): The offset value for pagination. Defaults to 0.
            limit (int, optional): The maximum number of documents to retrieve. Defaults to 10.

        Returns:
            list[dict[str, any]]: A list of documents matching the query.

        Raises:
            Exception: If an error occurs during the document search operation.
        """
        try:
            self.collection.load()

            search_params = {
                "metric_type": "L2", 
                "offset": offset, 
                "ignore_growing": False, 
                "params": {"nprobe": 10}
            }
  
            document = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=f'userId == "{user_id}"',
                output_fields=['details'],
                consistency_level="Strong",
                _async=True
            )

            results = document.result()

            self.collection.release()

            return results
    
        except Exception as e:
            print(f"An error occurred: {e}")

            return {"error": "An unexpected error occurred during the document search operation."}
        
    def update_user_document_by_id(self,
                                   user_id: str,
                                   file_id: str,
                                   node_id: str,
                                   document_embedding: list[float],
                                   details: Dict[str, Any]) -> str:
        """
        Updates a document in the Milvus collection based on its ID.

        Args:
            userId (str): The ID of the user who owns the document.
            fileId (str): The ID of the file associated with the document.
            nodeId (str): The ID of the text node associated with the document.
            document_embedding (list[float]): The embedding vector representation of the document.
            details (Dict[str, Any]): Additional details or metadata of the document.

        Returns:
            str: The primary key of the updated document.

        Raises:
            Exception: If an error occurs during the document update operation.
        """
        try:
            self.collection.load()

            document = {
                "userId": user_id,
                "fileId": file_id,
                "textNodeId": node_id,
                "embedding": document_embedding,
                "details": details
            }
            
            result = self.collection.upsert(data=document)

            index_params = {
                "index_type": "IVF_FLAT",  # index type
                "metric_type": "L2",  # distance metric
                "params": {"nlist": 1024}  # index params
            }

            self.collection.create_index(field_name="embedding", index_params=index_params)

            self.collection.release()

            return result.primary_keys
            
        except Exception as e:
            print(f"An error occurred: {e}")

            return {"error": "An unexpected error occurred during the document update operation."}
            
    def get_user_document_by_file_id(self, user_id: str, file_id: str) -> Dict[str, Any]:
        """
        Retrieves a document from the Milvus collection based on its user ID and file ID.

        Args:
            userId (str): The ID of the user who owns the document.
            fileId (str): The ID of the file associated with the document.

        Returns:
            Dict[str, Any]: The details of the retrieved document.

        Raises:
            Exception: If an error occurs during the document retrieval operation.
        """
        try:
            self.collection.load()

            results = self.collection.query(
                expr=f'userId == "{user_id}" && fileId == "{file_id}"',
                output_fields=['textNodeId', 'details'],
                consistency_level="Strong",
                _async=True
            )

            self.collection.release()

            return results[0]
    
        except Exception as e:
            print(f"An error occurred: {e}")

            return {"error": "An unexpected error occurred during the document retrieval operation."}
                                  