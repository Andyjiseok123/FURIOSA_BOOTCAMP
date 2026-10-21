#tool calling : 외부 기능 가져와서 사용하기!!!
#lg 05 copy

from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1"

llm = ChatOpenAI(
    model_name = 'gpt-5-nano',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url
)


################################## 0. Tool 정의 ##################################
# pip install langchain_tavily
from langchain_tavily import TavilySearch

#웹 검색 동구 생성
web_search = TavilySearch(
    max_results = 3,
)

# result = web_search.invoke({
#     "query" : "삼성전자의 오늘 주가 알려줘"
# })
# print(result)
# print(result['results'][-1]['content'])

tools = [web_search]

llm_with_tool = llm.bind_tools(tools)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

######################## 1. State 정의 ###################################
class State(TypedDict):
    messages : Annotated[list, add_messages]


######################## 2. 노드 정의 ###################################
def chatbot(state : State):
    return {"messages" : [llm_with_tool.invoke(state["messages"])]}

######################## 3. 그래프 구성 ####################################
# 그래프 생성
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)

tool_node = ToolNode(tools)
graph_builder.add_node("tools",tool_node)

######################## 4. 엣지 추가 ######################################
graph_builder.add_edge(START,"chatbot")
# graph_builder.add_edge("chatbot","tools")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition     #tool_calls가 있냐?
    # tool_calls가 있으면 'tools' 반환
    # tool_calls가 없으면 END 반환
    # tools_condtion에 END가 포함되어 있음.
)
graph_builder.add_edge("tools","chatbot")

######################## 5. 컴파일 #####################################
graph = graph_builder.compile()

######################## 6. 그래프 실행 #################################
# query = "삼성전자의 오늘 주가를 알려줘"

# response = graph.invoke({
#     "messages" : [("user", query)]
# })

# print("AI : ", response["messages"][-1].content)

##########################################################################
# 그래프 이미지 저장
##########################################################################
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

png_data = graph.get_graph().draw_mermaid_png()
# with open(path + "lg07_ToolNode.png",'wb') as file:
#     file.write(png_data)
#     #with는 자동 close()
img = Image.open(BytesIO(png_data))

# 그림 출력
plt.figure(figsize=(12,8))
plt.imshow(img)
plt.axis('off')
plt.show()
