"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8
furiosa-llm serve furiosa-ai/gpt-oss-20b
furiosa-llm serve furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic

furiosa-llm serve furiosa-ai/Qwen3-4B-FP8\
    --devices npu:3 \
    --host 0.0.0.0 \
    --port 8000

"""

from openai import OpenAI
from langchain_openai import ChatOpenAI

# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="EMPTY"
# )
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"
)


# 서버에 로드된 모델 확인
# models = client.models.list()
# model_id = models.data[0].id

# print("모델 :",model_id)

# # 추론
# response = client.chat.completions.create(
#     model = model_id,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "NPU를 활용한 physical AI 프로젝트 추천해줘."
#         }
#     ],
#     max_tokens=1500
# )

response = llm.invoke("NPU를 활용한 physical AI 프로젝트 추천해줘.")


# print(response.choices[0].message.content)

print(response.content)