from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
import openai


openai.api_key = "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"

# load documents
documents = SimpleDirectoryReader("llama_index/data").load_data()

print("Document ID:", documents[0].doc_id)

vector_store = MilvusVectorStore(dim=1536, overwrite=True, host="localhost", port=19530)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
# Either way we can now query the index
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")

print(response)

