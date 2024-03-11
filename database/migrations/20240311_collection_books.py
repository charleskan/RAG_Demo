# client = MilvusClient(
# uri='http://localhost:19530',
# token='root:Milvus',
# )

# connections.add_connection(
#   default={"host": "localhost", "port": "19530"}
# )

import os
from dotenv import load_dotenv

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

load_dotenv()

# Connect to Milvus
connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))

id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
chunk_metadata = FieldSchema(name="chunk_metadata", dtype=DataType.VARCHAR, max_length=1280)

fields = [id, embedding, chunk_metadata]

schema = CollectionSchema(fields=fields)

collection_name = "book03"

collection = Collection(name=collection_name, schema=schema)

print(f"Collection {collection_name} created successfully.")