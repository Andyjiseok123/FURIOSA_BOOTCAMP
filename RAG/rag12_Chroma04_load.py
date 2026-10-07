import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from glob import glob

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
    dimensions=5,
)

DB_PATH = "./_db/Chroma12/"

vector_store = Chroma(
    # documents=texts,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12",
)
print(vector_store._collection.count())

