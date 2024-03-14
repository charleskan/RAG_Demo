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

    query_embedding = Settings.embed_model.get_text_embedding("測試的書作者是誰？")

    import numpy as np

    # calculate the L2 norm of the query vector
    norm = np.linalg.norm(query_embedding)

    # check if the norm is close to 1
    if np.isclose(norm, 1):
        print("vector is normalized.")
    else:
        print("vector is not normalized.")

    repository = MilvusDocumentRepository()


    getResults = repository.get_document_by_fileId(userId="1", fileId="1")
    print(f"get: {getResults}")

    data_json01 = {
        "details": {
            "metadata": {
                "title": "測試的書",
                "author": "YOYOYO",
                "publication_date": "2022",
            },
            "context": "這是一本專門用來測試的書，作者是YOYOYO，出版日期是2044年。"
        }
    }

    updateResults = repository.update_document_by_id(
                                                userId="1", 
                                               fileId="1", 
                                               nodeId="1", 
                                               document_embedding=query_embedding, 
                                               details=data_json01)
    print(f"update: {updateResults}")

    searchResults = repository.get_document_by_query_embedding(userId="1", query_embedding=query_embedding)
    print(f"search: {searchResults}")

    collection.release()


if __name__ == "__main__":
    asyncio.run(main())