import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.vectorstores import FAISS
from glob import glob

from dotenv import load_dotenv
load_dotenv()


api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=api_key,
#     base_url=base_url,
# )
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={
#         "device" : "cpu",           #GPU로 하고싶을때는 cuda 다만 pytorch와 맞는 버전의 cuda가 깔려있어야함
#         "local_files_only" : True
#     }
# )

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device" : "cpu",           #GPU로 하고싶을때는 cuda 다만 pytorch와 맞는 버전의 cuda가 깔려있어야함
        "local_files_only" : True #한번 parameter를 받으면 이 옵션을 켜서 실행시간 단축 가능
    }
)

DB_PATH = "./_db/Faiss19/"

vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index19',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    )

# query = "삼성전자의 창업주는 누구인가요?"
retriever = vector_store.as_retriever(
    serch_type="similarity_score_threshold",
    search_kwarges={
        "k":2,
        "score_threshold":0.7 # 유사도가 0.7 이하일 경우 검색에서 제외
    }
)
# aaa = retriever.invoke(query)

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gemini-3.5-flash",
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

from langchain_core.prompts import MessagesPlaceholder      #프롬프트에 history를 넣을 공간 생성
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages([
    ("system","""
    당신은 강력한 AI 챗봇입니다.

    다음 규칙에 따라 답변하세요.
    1. 이전대화에서 확인할 수 있는 정보는 이전 대화를 참고하여 답변하세요.
    2. 일반적인 인사나 일상 대화는 자연스럽게 답변하세요.
    3. 문서에 관한 질문은 제공된 컨텍스트를 근거로 답변하세요.
    4. 문서에 관한 질문인데 컨텍스트에 관련 정보가 없다면
    "주어진 정보로 답변할 수 없습니다." 라고 답변하세요.

    컨텍스트 : {context}
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human","{input}")
])

# 체인 만들기
docu_chain = create_stuff_documents_chain(model, prompt)        # prompt | model 과 같은 느낌
rag_chain = create_retrieval_chain(retriever, docu_chain)       # 검색(vectorDB에 있는 정보) | docu_chain
# retriever가 vectorDB에서 검색한 내용을 context로 반환해줌

"""
# 체인 실행
query = "삼성전자의 창업자는 누구인가요?"
response = rag_chain.invoke({"input" : query})

print(response)
#vectorDB내에 관련 정보가 없어서 "주어진 정보로는 답변할 수 없습니다" 라고 나옴
print(response.keys())
# dict_keys(['input','context','answer'])
"""

################################### Gradio 챗봇 ###########################################
import gradio as gr

def answer_invoke(message, history):
    chat_history = []

    for item in history:
        content = item["content"]

        if item["role"] =="user":
            chat_history.append(HumanMessage(content=content))
        elif item["role"] == "assistant":
            chat_history.append(AIMessage(content=content))


            
    response = rag_chain.invoke({
        "input" : message,
        "chat_history" : chat_history
        })

    # 터미널 상으로 실제 검색된 문서 2개 확인.  [a,b,c]
    for i, doc in enumerate(response["context"], start=1):
        print(f"\n검색 문서 {i}")
        print("출처: ", doc.metadata.get("source"))
        print("내용: ", doc.page_content)
        print("======================================================")

    
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='테스트봇!!')

# Gradio 실행
demo.launch()

"""
1.안녕이라고 쳤을때
history = [
    {"role" : "user", "content" : "안녕"},
    {"role" : "assistant", "content" : "안녕하세요"},
]

2.나는 이지석이다 라고 쳤을때
history = [
    {"role" : "user", "content" : "안녕"},
    {"role" : "assistant", "content" : "안녕하세요"},
    {"role" : "user", "content" : "나는 이지석이다"},
    {"role" : "assistant", "content" : "이지석님 이시군요"},
]

"""