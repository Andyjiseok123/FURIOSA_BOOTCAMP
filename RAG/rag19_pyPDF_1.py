from langchain_community.document_loaders import TextLoader, PyPDFLoader

path = './_data/'
file_name = 'Attention is all you need.pdf'
pdf_loader = PyPDFLoader(path + file_name)
pdf_docs = pdf_loader.load()

# print(type(pdf_docs))           # <class 'list'>
# print(len(pdf_docs))            # 15
# print(pdf_docs)

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# FIASS는 유클리드 거리 기반 유사도 사용

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

#1. 데이터 불러오기
# loader2 = TextLoader(path + "nvidia_outlook.txt",encoding='utf-8')

#2. 데이터 자르기.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n","\n"," ",""],
)

split_pdf = text_splitter.split_documents(pdf_docs)

# print(split_doc1)
# print(len(split_doc1),len(split_doc2))      #9 9

#3. 임베딩
from langchain_openai.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

########################## 여기부터 faiss #################################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
# faiss_index = faiss.IndexFlatL2(1536)
# print("FAISS 인덱스 초기화 준비 완료")

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
# print(faiss_index.d)        # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# 저장된 문서의 갯수 확인
print(faiss_db.index.ntotal)        # 0

####################### 준비 완료 #######################
#######################################################

db = FAISS.from_documents(
    documents=split_pdf,
    embedding=embeddings,
)

DB_PATH = './_db/Faiss19'
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index19'
)

# faiss_index17.faiss와 faiss_index17.pkl 파일 생성