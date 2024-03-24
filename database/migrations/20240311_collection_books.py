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

textNodeId = FieldSchema(name="textNodeId", dtype=DataType.VARCHAR, max_length=1280, is_primary=True)
userId = FieldSchema(name="userId", dtype=DataType.VARCHAR, max_length=128)
fileId = FieldSchema(name="fileId", dtype=DataType.VARCHAR, max_length=128)
embedding = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
document = FieldSchema(name="document", dtype=DataType.JSON)

fields = [userId, fileId, textNodeId, embedding, document]

schema = CollectionSchema(fields=fields)

collection_name = "book03"

collection = Collection(name=collection_name, schema=schema)

print(f"Collection {collection_name} created successfully.")
