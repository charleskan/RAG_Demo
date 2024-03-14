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

import os

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
