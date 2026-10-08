# LCEL = LangChain Expression Language
# chain = prompt | model | output_parser

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate

# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.environ["MONOROUTER_API_KEY"].strip()
# base_url = "https://monogpt.kr/api/monorouter/v1"


# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=api_key,
#     base_url=base_url
# )

# pip langchain_huggingface.embeddings
# pip install sentence-transformers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "device" : "cpu",           #GPU로 하고싶을때는 cuda 다만 pytorch와 맞는 버전의 cuda가 깔려있어야함
        # "local_files_only" : True #한번 parameter를 받으면 이 옵션을 켜서 실행시간 단축 가능
    }
)

prompt = "삼성전자의 창업주는 누구인가요?"

vector = embeddings.embed_query(prompt)
print(vector)
print("===============================================================")
print("임베딩 벡터의 차원 : ",len(vector))          #1024