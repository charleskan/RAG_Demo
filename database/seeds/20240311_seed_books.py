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
        "metadata": {
            "title": "測試的書",
            "author": "abc",
            "publication_date": "2022",
        },
        "content": "這是一本專門用來測試的書，作者是abc，出版日期是2022年。"
}
userId01 = "1"
fileId01 = "1"
nodeId01 = "1"
data_content01 = data_json01["content"]
data_embeddings01 = Settings.embed_model.get_text_embedding(data_content01)
document01 = {
    "userId": userId01,
    "fileId": fileId01,
    "textNodeId": nodeId01,
    "embedding": data_embeddings01,
    "document": data_json01
}
mr01 = collection.insert(data=document01)
print("Data inserted successfully. Primary keys of the inserted data: ", mr01.primary_keys)

#02
data_json02 = {
        "metadata": {
            "title": "testing book",
            "author": "Charles",
            "publication_date": "2033",
            
        },
        "content": "this is a book for testing, the author is Charles, and the publication date is 2033."
}
userId02 = "1"
fileId02 = "2"
nodeId02 = "2"
data_content02 = data_json02["content"]
data_embeddings02 = Settings.embed_model.get_text_embedding(data_content02)
document02 = {
    "userId": userId02,
    "fileId": fileId02,
    "textNodeId": nodeId02,
    "embedding": data_embeddings02,
    "document": data_json02
}
mr02 = collection.insert(data=document02)
print("Data inserted successfully. Primary keys of the inserted data: ", mr02.primary_keys)

#03
data_json03 = {
        "metadata": {
            "title": "Yu-Gi-Oh!",
            "author": "Sean Kan",
            "publication_date": "2043",
        
        },
        "content": "Yu-Gi-Oh! is a great anime, I like it very much."
}
userId03 = "1"
fileId03 = "3"
nodeId03 = "3"
data_content03 = data_json03["content"]
data_embeddings03 = Settings.embed_model.get_text_embedding(data_content03)
document03 = {
    "userId": userId03,
    "fileId": fileId03,
    "textNodeId": nodeId03,
    "embedding": data_embeddings03,
    "document": data_json03
}
mr03 = collection.insert(data=document03)
print("Data inserted successfully. Primary keys of the inserted data: ", mr03.primary_keys)

#04
data_json04 = {
        "metadata": {
            "title": "Yu-Gi-Oh!",
            "author": "Charles Kan",
            "publication_date": "2033",
        },
        "content": "Yu-Gi-Oh! is a great anime. Author is Charles Kan, publication date is 2033"
}
userId04 = "1"
fileId04 = "4"
nodeId04 = "4"
data_content04 = data_json04["content"]
data_embeddings04 = Settings.embed_model.get_text_embedding(data_content04)
document04 = {
    "userId": userId04,
    "fileId": fileId04,
    "textNodeId": nodeId04,
    "embedding": data_embeddings04,
    "document": data_json04
}
mr04 = collection.insert(data=document04)
print("Data inserted successfully. Primary keys of the inserted data: ", mr04.primary_keys)

#05
data_json05 = {
        "metadata": {
            "title": "遊戲攻略書",
            "author": "Sean Kan",
            "publication_date": "2043",
        },
        "content": "遊戲攻略書是一種很特別的書，它可以幫助你通過遊戲。作者是Sean Kan，出版日期是2043年。"
}
userId05 = "1"
fileId05 = "5"
nodeId05 = "5"
data_content05 = data_json05["content"]
data_embeddings05 = Settings.embed_model.get_text_embedding(data_content05)
document05 = {
    "userId": userId05,
    "fileId": fileId05,
    "textNodeId": nodeId05,
    "embedding": data_embeddings05,
    "document": data_json05
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