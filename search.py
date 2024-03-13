import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
from llama_index.core import (Settings)
import asyncio


from src.feature.document.data.repositories.document_repository import MilvusDocumentRepository

async def main():

    load_dotenv()

    connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))

    collection = Collection("book03")      # Get an existing collection.
    collection.load()

    search_params = {
        "metric_type": "L2", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }

    query_embedding = Settings.embed_model.get_text_embedding("what is Yu-Gi-Oh? who is the author? when was it published?")

    import numpy as np

    # calculate the L2 norm of the query vector
    norm = np.linalg.norm(query_embedding)

    # check if the norm is close to 1
    if np.isclose(norm, 1):
        print("vector is normalized.")
    else:
        print("vector is not normalized.")

    repository = MilvusDocumentRepository()

    results = repository.get_document_by_query_embedding(query_embedding)
    # get the IDs of all returned hits
    results[0].ids
    # get the distances to the query vector from all returned hits
    print(results[0].distances)
    # get the value of an output field specified in the search request.
    hit = results[0][0]
    metadata = hit.entity.get('details')
    print(metadata)
    context = metadata["details"]["context"]
    print(context)

    collection.release()


if __name__ == "__main__":
    asyncio.run(main())