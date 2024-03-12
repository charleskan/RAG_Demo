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
        "context": "這是一本專門用來測試的書，作者是abc，出版日期是2022年。"
    }
}
data_context01 = data_json01["metadata"]["context"]
data_embeddings01 = Settings.embed_model.get_text_embedding(data_context01)
mr01 = collection.insert([[data_embeddings01], [data_json01]])
print("Data inserted successfully. Primary keys of the inserted data: ", mr01.primary_keys)

#02
data_json02 = {
    "metadata": {
        "title": "testing book",
        "author": "Charles",
        "publication_date": "2033",
        "context": "this is a book for testing, the author is Charles, and the publication date is 2033."
    }
}
data_context02 = data_json02["metadata"]["context"]
data_embeddings02 = Settings.embed_model.get_text_embedding(data_context02)
mr02 = collection.insert([[data_embeddings02], [data_json02]])
print("Data inserted successfully. Primary keys of the inserted data: ", mr02.primary_keys)

#03
data_json03 = {
    "metadata": {
        "title": "遊戲王",
        "author": "Sean Kan",
        "publication_date": "2043",
        "context": "遊戲王是一部很棒的動畫，我很喜歡。"
    }
}
data_context03 = data_json03["metadata"]["context"]
data_embeddings03 = Settings.embed_model.get_text_embedding(data_context03)
mr03 = collection.insert([[data_embeddings03], [data_json03]])
print("Data inserted successfully. Primary keys of the inserted data: ", mr03.primary_keys)

#04
data_json04 = {
    "metadata": {
        "title": "Yu-Gi-Oh!",
        "author": "Sean Kan",
        "publication_date": "2043",
        "context": "Yu-Gi-Oh! is a great anime, I like it very much."
    }
}
data_context04 = data_json04["metadata"]["context"]
data_embeddings04 = Settings.embed_model.get_text_embedding(data_context04)
mr04 = collection.insert([[data_embeddings04], [data_json04]])
print("Data inserted successfully. Primary keys of the inserted data: ", mr04.primary_keys)

#05
data_json05 = {
    "metadata": {
        "title": "遊戲攻略書",
        "author": "Sean Kan",
        "publication_date": "2043",
        "context": "遊戲攻略書是一種很特別的書，它可以幫助你通過遊戲。作者是Sean Kan，出版日期是2043年。"
    }
}
data_context05 = data_json05["metadata"]["context"]
data_embeddings05 = Settings.embed_model.get_text_embedding(data_context05)
mr05 = collection.insert([[data_embeddings05], [data_json05]])
print("Data inserted successfully. Primary keys of the inserted data: ", mr05.primary_keys)


index_params = {
    "index_type": "IVF_FLAT",  # 选择一个索引类型
    "metric_type": "L2",  # 指定距离计算类型，L2是欧氏距离
    "params": {"nlist": 1024}  # 索引参数，nlist值根据数据量和查询需求调整
}

collection.create_index(field_name="embedding", index_params=index_params)

print("Index created for 'embedding'")