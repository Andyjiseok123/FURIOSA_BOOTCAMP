from openai import OpenAI
from langchain_openai import ChatOpenAI
import time
max_tokens = 1000

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    max_tokens = max_tokens,
    temperature=0,
    top_p=0.9                   #이게 뭔지 공부
)
start = time.perf_counter()
response = llm.invoke("NPU를 활용한 physical AI 프로젝트 추천해줘.")
end = time.perf_counter()
elapsed = round(end-start,2)


# print(response.content)
print("추론시간 : ", elapsed, "초")
print("출력토큰 : ",max_tokens)
print("평균 처리량 : ", round(max_tokens/elapsed ,2), "tokens/s")
