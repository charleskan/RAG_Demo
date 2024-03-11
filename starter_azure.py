from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core import Settings


api_key = "e4a079b894954fce98e467b9fde69dc3"
azure_endpoint = "https://angus-vision-gpt-aiservices1492894653.openai.azure.com/openai/deployments/angus-gpt-4-vision/chat/completions?api-version=2023-12-01-preview"
api_version = "2022-12-01-preview"

llm = AzureOpenAI(
    model="text-davinci-003",
    deployment_name="angus-gpt-4-vision",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
# embed_model = AzureOpenAIEmbedding(
#     model="text-embedding-ada-002",
#     deployment_name="text-embedding-ada-002",
#     api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     api_version=api_version,
# )

embed_model = "local:BAAI/bge-small-en-v1.5"


Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(
    input_files=["data/paul_graham_essay.txt"]
).load_data()


index = VectorStoreIndex.from_documents(documents)

query = "What is most interesting about this essay?"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)