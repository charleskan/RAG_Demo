import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
import openai
from llama_index.core import (Settings)

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

openai.api_key = os.getenv('OPENAI_API_KEY')

query_embedding = Settings.embed_model.get_text_embedding("遊戲王是什麼？")

# print(f"Query embedding: {query_embedding}")

results = collection.search(
    data=[query_embedding], 
    anns_field="embedding", 
    # the sum of `offset` in `param` and `limit` 
    # should be less than 16384.
    param=search_params,
    limit=10,
    expr=None,
    # set the names of the fields you want to 
    # retrieve from the search result.
    output_fields=['chunk_metadata'],
    consistency_level="Strong"
)

# get the IDs of all returned hits
results[0].ids

# get the distances to the query vector from all returned hits
print(results[0].distances)

# get the value of an output field specified in the search request.
hit = results[0][0]
print(hit.entity.get('chunk_metadata'))

collection.release()