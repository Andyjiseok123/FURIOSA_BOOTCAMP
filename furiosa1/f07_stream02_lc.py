from langchain_openai import ChatOpenAI
from openai import OpenAI

# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="EMPTY"
# )

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    max_tokens = 1000,
    temperature=0,
    top_p=0.9
)

#스트리밍
# # 서버에 로드된 모델 확인
# model = client.models.list().data[0].id
# # model_id = models.data[0].id

# # print("모델 :",model_id)

# # 추론
# streaming = client.chat.completions.create(
#     model = model,
#     messages=[
#         {
#             "role" : "user",
#             "content" : "NPU를 활용한 physical AI 프로젝트 추천해줘."
#         }
#     ],
#     max_tokens=1500,
#     stream = True
# )

# # print(response.choices[0].message.content)

# for chunk in streaming:
#     if chunk.choices:
#         content = chunk.choices[0].delta.content
#         if content:
#             print(content, end="", flush=True)

for chunk in llm.stream("NPU를 활용한 physical AI 프로젝트 추천해줘."):
    print(chunk.content,end="",flush=True)