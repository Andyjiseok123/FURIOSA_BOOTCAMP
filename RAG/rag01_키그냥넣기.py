from langchain_openai import ChatOpenAI

openai_api_key = " "


llm = ChatOpenAI(
    model_name = 'gpt-5.6-terra',
    temperature=0,
    openai_api_key = openai_api_key,
)

response = llm.invoke('안녕하세요.')
"""
현재 메모리기능이 없어서 단순히 추론만 가능한상태
"""
# print(response)
print(response.content)