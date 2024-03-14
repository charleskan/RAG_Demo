from typing import Any, Dict
from pymilvus import connections, Collection
import os
import json
import asyncio


from src.feature.document.domain.repositories.i_document_repository import IDocumentRepository

class MilvusDocumentRepository(IDocumentRepository):

    def __init__(self):
        connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))
        self.collection = Collection("book03")

    def save_document(self, 
                      userId: str, 
                      fileId: str, 
                      nodeId: str, 
                      document_embedding: list[float], 
                      details: Dict[str, Any]) -> str:
        try:
            self.collection.load()

            document = {
                "userId": userId,
                "fileId": fileId,
                "textNodeId": nodeId,
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

    
    def get_document_by_query_embedding(self,
                                        userId: str, 
                                        query_embedding: list[float],
                                        offset: int = 0,
                                        limit: int = 10) -> list[dict[str, any]]:
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
                expr=f'userId == "{userId}"',
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
        
    def update_document_by_id(self,
                                    # id: str,
                                    userId: str, 
                                    fileId: str, 
                                    nodeId: str, 
                                    document_embedding: list[float], 
                                    details: Dict[str, Any]) -> str:
            try:
                self.collection.load()

                document = {
                    "userId": userId,
                    "fileId": fileId,
                    "textNodeId": nodeId,
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
            
    def get_document_by_fileId(self, userId: str, fileId: str) -> Dict[str, Any]:
        try:
            self.collection.load()

            results = self.collection.query(
                expr=f'userId == "{userId}" && fileId == "{fileId}"',
                output_fields=['textNodeId', 'details'],
                consistency_level="Strong",
                _async=True
            )

            self.collection.release()

            return results[0]
    
        except Exception as e:
            print(f"An error occurred: {e}")

            return {"error": "An unexpected error occurred during the document retrieval operation."}
                                  