import os
from dotenv import load_dotenv
from llama_index.core import (Settings)

import openai
from pymilvus import Collection, connections
import json

load_dotenv()

connections.connect(alias="default", host=os.getenv('MILVUS_HOST'), port=os.getenv("MILVUS_PORT"))

collection = Collection("book03")

# Insert data

#01
data_json01 = {
    "details": {
        "metadata": {
            "title": "測試的書",
            "author": "abc",
            "publication_date": "2022",
        },
        "context": "這是一本專門用來測試的書，作者是abc，出版日期是2022年。"
    }
}
nodeId01 = "1"
data_context01 = data_json01["details"]["context"]
data_embeddings01 = Settings.embed_model.get_text_embedding(data_context01)
document01 = {
    "textNodeId": nodeId01,
    "embedding": data_embeddings01,
    "details": data_json01
}
mr01 = collection.insert(data=document01)
print("Data inserted successfully. Primary keys of the inserted data: ", mr01.primary_keys)

#02
data_json02 = {
    "details": {
        "metadata": {
            "title": "testing book",
            "author": "Charles",
            "publication_date": "2033",
            
        },
        "context": "this is a book for testing, the author is Charles, and the publication date is 2033."
    }
}
nodeId02 = "2"
data_context02 = data_json02["details"]["context"]
data_embeddings02 = Settings.embed_model.get_text_embedding(data_context02)
document02 = {
    "textNodeId": nodeId02,
    "embedding": data_embeddings02,
    "details": data_json02
}
mr02 = collection.insert(data=document02)
print("Data inserted successfully. Primary keys of the inserted data: ", mr02.primary_keys)

#03
data_json03 = {
    "details": {
        "metadata": {
            "title": "Yu-Gi-Oh!",
            "author": "Sean Kan",
            "publication_date": "2043",
        
        },
        "context": "Yu-Gi-Oh! is a great anime, I like it very much."
    }
}
nodeId03 = "3"
data_context03 = data_json03["details"]["context"]
data_embeddings03 = Settings.embed_model.get_text_embedding(data_context03)
document03 = {
    "textNodeId": nodeId03,
    "embedding": data_embeddings03,
    "details": data_json03
}
mr03 = collection.insert(data=document03)
print("Data inserted successfully. Primary keys of the inserted data: ", mr03.primary_keys)

#04
data_json04 = {
    "details": {
        "metadata": {
            "title": "Yu-Gi-Oh!",
            "author": "Charles Kan",
            "publication_date": "2033",
        },
        "context": "Yu-Gi-Oh! is a great anime. Author is Charles Kan, publication date is 2033"
    }
}
nodeId04 = "4"
data_context04 = data_json04["details"]["context"]
data_embeddings04 = Settings.embed_model.get_text_embedding(data_context04)
document04 = {
    "textNodeId": nodeId04,
    "embedding": data_embeddings04,
    "details": data_json04
}
mr04 = collection.insert(data=document04)
print("Data inserted successfully. Primary keys of the inserted data: ", mr04.primary_keys)

#05
data_json05 = {
    "details": {
        "metadata": {
            "title": "遊戲攻略書",
            "author": "Sean Kan",
            "publication_date": "2043",
        },
        "context": "遊戲攻略書是一種很特別的書，它可以幫助你通過遊戲。作者是Sean Kan，出版日期是2043年。"
    }
}
nodeId05 = "5"
data_context05 = data_json05["details"]["context"]
data_embeddings05 = Settings.embed_model.get_text_embedding(data_context05)
document05 = {
    "textNodeId": nodeId05,
    "embedding": data_embeddings05,
    "details": data_json05
}
mr05 = collection.insert(data=document05)
print("Data inserted successfully. Primary keys of the inserted data: ", mr05.primary_keys)


index_params = {
    "index_type": "IVF_FLAT",  # index type
    "metric_type": "L2",  # distance metric
    "params": {"nlist": 1024}  # index params
}

collection.create_index(field_name="embedding", index_params=index_params)

print("Index created for 'embedding'")