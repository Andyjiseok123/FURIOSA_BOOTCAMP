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

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

################### 1. State 정의 ########################
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int
#위에꺼는 한마디로 아래다!!!!
# state = {
#     "messages" : [],
#     'aaa' : 'abc'
# }

################### 2. node 정의 #########################
def prepare(state : State):
    return {"messages" : 
            [AIMessage(content = "질문을 접수했습니다.")]
            }

def chatbot(state : State):
    return {"messages" : [llm.invoke(state["messages"])]}

def finish(state : State):
    return {"messages" : 
        [AIMessage(content = "답변 처리가 완료되었습니다.")]
        }
################### 3. 그래프 구성 ########################
graph_01 = StateGraph(State)

graph_01.add_node("prepare", prepare)
graph_01.add_node("chatbot", chatbot)
graph_01.add_node("finish", finish)

################### 4. 엣지 추가 ########################
graph_01.add_edge(START, "prepare")
graph_01.add_edge("prepare","chatbot")
graph_01.add_edge("chatbot","finish")
graph_01.add_edge("finish", END)

################### 5. 그래프 컴파일 ########################
graph = graph_01.compile()

################### 6. 그래프 실행 ########################
query = "삼성전자의 주가를 예측해줘"

response = graph.invoke({
    "messages" : [("user",query)]
})

################### 7. 결과 출력 ################################
for i,msg in enumerate(response["messages"]):
    print(f"[{i}] {msg.type} : {msg.content}")

"""
[0] human : 삼성전자의 주가를 예측해줘
[1] ai : 질문을 접수했습니다.
[2] ai : 주가를 정확히 예측할 수는 없습니다. 다만 삼성전자 stock에 대해 도움이 될 수 있는 정보를 정리해 드릴 수는 있습니다.

- 현재 상황 요약: 실시간 주가나 최신 뉴스는 시세 서비스나 금융 포털에서 확인해야 합니다. 예를 들어 증권사 앱, 네이버 금융, Yahoo Finance 등에서 "005930" 종목 코드를 검색하면 현재가, 시가, 고가/저가, 거래량, 체결가 등을 확인할 수 있습니다.
- 영향 요인: 기업 실적(매출/영업이익/순이익), 분기/연간 가이던스, 반도체 경기 사이클, 전체 IT 수요, 환율, 금리, 글로벌 공급망 이슈, 배당 정책 등.
- 예측 접근 방법:
  - 기본적 분석: 최근 실적 발표, 경쟁사 비교, 시장 점유율, 신제품/라인업, R&D 투자.
  - 기술적 분석: 이동평균선, RSI, MACD 등 차트 신호를 통해 단기 트렌드 확인.
  - 이벤트 기반: 반도체 공급망 이슈, 반도체 가격, 반도체 수요 추이, 미국/중국 정책.
- 리스크 관리 팁:
  - 분산 투자, 손절/목표가 설정, 투자 금액 한도 설정.
  - 장기 투자 시 기업의 펀더멘털과 현금흐름에 집중.

도움이 필요하신 부분을 말씀해 주시면, 예측 모델 접근 방법이나 실시간 데이터 확인 방법, 체크리스트를 구체적으로 제공해 드리겠습니다. 예를 들어:
- 최근 분기 실적 요약과 해석
- 주요 설비 투자나 신제품 소식
- 기술적 차트의 간단한 분석 방법

또는 원하시면 제가 삼성전자에 대한 기본적 모의 분석(가정 시나리오 기반)이나 체크리스트를 만들어 드릴게요.
[3] ai : 답변 처리가 완료되었습니다.
"""
