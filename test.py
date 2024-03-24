import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
from llama_index.core import (Settings)
import asyncio
import numpy as np


from src.feature.document.data.repositories.document_repository import MilvusDocumentRepository

async def main():

    load_dotenv()

    query_embedding = Settings.embed_model.get_text_embedding("測試的書作者是誰？")

    # calculate the L2 norm of the query vector
    norm = np.linalg.norm(query_embedding)

    # check if the norm is close to 1
    if np.isclose(norm, 1):
        print("vector is normalized.")
    else:
        print("vector is not normalized.")

    repository = MilvusDocumentRepository()


    getResults = repository.get_user_document_by_file_id(user_id="1", file_id="1")
    print(f"get: {getResults}")

    data_json01 = {
            "metadata": {
                "title": "測試的書",
                "author": "YOYOYO",
                "publication_date": "2022",
            },
            "content": "這是一本專門用來測試的書，作者是YOYOYO，出版日期是2044年。"
    }

    updateResults = repository.update_user_document_by_id(user_id="1",
                                                          file_id="1",
                                                          node_id="1",
                                                          document_embedding=query_embedding,
                                                          details=data_json01)
    print(f"update: {updateResults}")

    searchResults = repository.search_user_documents_by_embedding(user_id="1", embedding=query_embedding)

    print(f"search: {searchResults[0].content}")

    # collection.release()


if __name__ == "__main__":
    asyncio.run(main())