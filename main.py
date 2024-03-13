from flask import Flask
from flask import request

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama

import openai

from llama_index.llms.openai import OpenAI

import os
from dotenv import load_dotenv

openai.api_key = os.getenv('OPENAI_API_KEY')

book_intro_embedding = Settings.embed_model.get_text_embedding("我是一本專門用來測試的書，作者是abc，出版日期是2022年。")

print(book_intro_embedding)


from pymilvus import Collection, MetricType, SearchParam

def search_embedding(collection_name, query_embedding, top_k=1):
    collection = Collection(collection_name)
    search_params = {"metric_type": MetricType.L2, "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding], 
        anns_field="book_intro", 
        limit=top_k, 
        expr=None
    )
    return results


app = Flask(__name__)

@app.route('/')
def hello_world():
    return "FlaskAPI is running"

@app.route("/query", methods=["POST"])
def query():
    query_text = request.args.get("text", None)
    if query_text is None:
        return (
            "No text found, please include a ?text=blah parameter in the URL",
            400,
        )
    resp = Settings.llm.complete(query_text)
    return str(resp), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
