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
)

DB_PATH = "./_db/Chroma12/"

vector_store = Chroma(
    # documents=texts,
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name="chroma12",
)

query = "삼성전자의 창업주는 누구인가요?"
retriever = vector_store.as_retriever(search_kwarges={"k":2})
aaa = retriever.invoke(query)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5.6-terra",
    temperature=0, #0:있는 그대로, 1:창의적으로
    max_completion_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# query_with_context=f"""
# {aaa[0].page_content}\n\n
# 위 내용을 근거하여 다음 질문에 답변하세요.\n\n{query}
# """

# response = model.invoke(query_with_context)
# print(response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요.

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)        # prompt | model 과 같은 느낌
rag_chain = create_retrieval_chain(retriever, docu_chain)       # 검색(vectorDB에 있는 정보) | docu_chain
# retriever가 vectorDB에서 검색한 내용을 context로 반환해줌
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
#vectorDB내에 관련 정보가 없어서 "주어진 정보로는 답변할 수 없습니다" 라고 나옴
print(response.keys())
# dict_keys(['input','context','answer'])