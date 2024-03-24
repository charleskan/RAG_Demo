# RAG_Demo

# Getting Started
```
pip install python-dotenv
pip3 install pymilvus
pip install llama_index.core
pip install llama-index-embeddings-openai
```

This project uses OpenAI's gpt-3.5-turbo by default for the embeddings of the text.

Make sure your API key is available to your code by setting it as an environment variable. In MacOS and Linux, this is the command:
```
export OPENAI_API_KEY=XXXXX
```

In Windows, you can set the environment variable using the following command:
```
set OPENAI_API_KEY=XXXXX
```

# Setting up the Milvus server
Then you should create .env file set the following environment variables for the Milvus server like this:
```
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

Then you can start the Milvus server by running the following command:
```
cd database
bash standalone_embed.sh start
```

# Migration and seeding
```
cd migrations
python3 20240311_collection_books.py
cd ..
```

```
cd seeds
python3 20240311_seed_books.py
cd ..
cd ..
```

# Running the tests
```
python3 test.py
```

# Usage
```
```