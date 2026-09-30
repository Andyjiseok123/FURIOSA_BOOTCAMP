from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = " "
# 어차피 휘발성이어서 한번 코드 실행 후 윗줄 삭제하고 실행하면 실행 안됨"

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('안녕하세요.')
# print(response)
print(response.content)