# 17-1 copy

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# FIASS는 유클리드 거리 기반 유사도 사용

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

#3. 임베딩
from langchain_openai.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Faiss17'
# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17'
# )
# faiss_index17.faiss와 faiss_index17.pkl 파일 생성

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("=================================")
#문서 저장소 ID 확인
print(db.index_to_docstore_id)
print("=================================")

#저장된 결과 확인
print(db.docstore._dict)
print("=================================")

#유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘",k=2)
print(aaa)
