from pymilvus import connections, Collection
import os
import json
from feature.document.domain.repositories.i_document_repository import IDocumentRepository

class MilvusDocumentRepository(IDocumentRepository):

    def __init__(self):
        connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))
        self.collection = Collection("book03")

    def save_document(self, 
            nodeId: str, 
            document_embedding: list[float], 
            details: dict[str, any]) -> dict[str, any]:
        document = {
            "textNodeId": nodeId,
            "embedding": document_embedding,
            "details": details
        }
        mr = self.collection.insert(data=document)
        return mr.primary_keys
    
    def get_document_by_query_embedding(self, 
                                       query_embedding: list[float],
                                       offset: int = 0,
                                       limit: int = 10,
                                       expression: str = None,
                                       metric_type: str = "L2",
                                       ignore_growing: bool = False,
                                       nprobe: int = 10,
                                       consistency_level: str = "Eventually") -> dict[str, any]:
        # """
        # :metric_type: This specifies the metric used to calculate the distance between vectors. L2 represents the Euclidean distance, also known as L2 distance. 
        # It's the most commonly used distance metric, calculating the straight-line distance between two points.

        # :offset: This parameter is used for paginated queries, specifying the starting point for returning results. 
        # 0 means that results start from the first item.

        # :ignore_growing: This parameter indicates whether the search operation should ignore data that is currently being inserted. 
        # False means the search considers all data, including the most recently inserted.

        # :params: This is a nested dictionary specifying more detailed search parameters. In this example, it contains a key-value pair "nprobe": 10.

        # :nprobe: This parameter is relevant for search algorithms that use indexing, such as the IVF (Inverted File) index. 
        # nprobe specifies the number of candidate lists to access during the search process. 
        # Increasing the nprobe value can improve the accuracy of the search but also increases the computational cost and time. 
        # In this example, nprobe is set to 10, meaning the search will consider 10 of the most likely candidate lists containing the query results.

        # In summary, the params dictionary within search_params is primarily used to refine the specific behaviors of the search operation, 
        # with nprobe being a key parameter for balancing search precision and performance. 
        # By adjusting these parameters, you can optimize the search operation's performance and accuracy according to your specific needs.
        # """
        search_params = {
            "metric_type": metric_type, 
            "offset": offset, 
            "ignore_growing": ignore_growing, 
            "params": {"nprobe": nprobe}
        }
        results = self.collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params,
            limit=limit,
            expr=expression,
            output_fields=['details'],
            consistency_level=consistency_level
        )
        return results