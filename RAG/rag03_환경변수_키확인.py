from langchain_openai import ChatOpenAI
import os

"""
admin 환경변수에 추가할 수 있음
"""

llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('안녕하세요.')
# print(response)
print(response.content)