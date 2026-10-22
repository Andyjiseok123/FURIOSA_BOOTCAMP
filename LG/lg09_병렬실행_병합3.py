from typing import TypedDict, Annotated
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

# 1. State 정의
class State(TypedDict):
    messages : Annotated[list, add_messages]

# 2. 검색 노드
def search_agent(state : State):
    print("검색 에이전트 실행")
    question = state["messages"][-1].content

    result = web_sea



# 3. 분석 노드
def analysis_agent(state : State):
    print("분석 에이전트 실행")
    return {"analysis_result" : "AI 반도체 수요 증가가 성장 요인이다."}

# 4. 결과 통합 노드
def merge_agent(state : State):
    result = (
        state["search_result"] + "\n"+
        state["analysis_result"]
    )
    return {"final_result" : result}

# 5. 그래프 구성
builder = StateGraph(State)

builder.add_node("search",search_agent)
builder.add_node("analysis",analysis_agent)
builder.add_node("merge",merge_agent)

############# 병렬 실행 #################
builder.add_edge(START,"search")
builder.add_edge(START,"analysis")

############# 병합 #################
builder.add_edge("search","merge")
builder.add_edge("analysis","merge")

builder.add_edge("merge",END)

graph = builder.compile()

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