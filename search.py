import os
from dotenv import load_dotenv
from pymilvus import Collection, connections
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

query_embedding = Settings.embed_model.get_text_embedding("遊戲王是什麼？")

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
    output_fields=['metadata'],
    consistency_level="Eventually"
)

# get the IDs of all returned hits
results[0].ids

# get the distances to the query vector from all returned hits
print(results[0].distances)

# get the value of an output field specified in the search request.
hit = results[0][0]
metadata = hit.entity.get('metadata')
print(metadata)
context = metadata["metadata"]["context"]
print(context)

collection.release()