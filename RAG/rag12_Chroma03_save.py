import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

path = './_data/rag_data/'

from glob import glob
# 폴더에서 텍스트 파일 목록 가져오기
txt_files = glob(os.path.join(path,'*.txt'))
print(txt_files)


#1. 데이터 불러오기
data=[]
for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    data += loader.load()

char_count = [len(doc.page_content) for doc in data]
# print(char_count)       # [8158, 2049, 1898]

#2. 데이터 자르기.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    separators=["\n\n","\n"," ",""],
)

texts = text_splitter.split_documents(data)
# print("생성된 텍스트 청크수 : ",len(texts))
# 생성된 텍스트 청크수 :  59
# print("각 청크의 길이 : ", list(len(text.page_content)for text in texts))
# 각 청크의 길이 :  [259, 154, 150, 282, 128, 276, 288, 258, 268, 262, 271, 226, 268, 182, 213, 281, 257, 182, 162, 
#              208, 259, 188, 225, 229, 219, 214, 258, 207, 284, 297, 198, 245, 177, 272, 215, 9, 269, 293, 290, 236, 289, 
#              209, 222, 230, 254, 249, 296, 181, 247, 243, 185, 219, 239, 235, 298, 299, 170, 187, 249]
# print("첫번째 청크의 내용 : ",texts[0].page_content)
# print("첫번째 청크의 길이 : ",len(texts[0].page_content))


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,
    base_url=base_url,
)

sample_text = "삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
# print(vector)


DB_PATH = './_db/Chroma12/'

#저장
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

print(f"벡터 저장소에 저장된 문서 수 : {vector_store._collection.count()}")
# 벡터 저장소에 저장된 문서 수 : 59
query = "삼성전자의 창업자는 누구인가요?"
result= vector_store.similarity_search(query)

print(f"검색 결과의 길이: {len(result)}")


############################# Retrieveres #############################
################################ 검색기 ################################
retriever = vector_store.as_retriever(search__kwargs={"k":2})
aaa = retriever.invoke(query)
print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}")
