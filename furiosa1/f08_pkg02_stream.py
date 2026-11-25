from furiosa1.a_kingkeras import get_llm

llm = get_llm()

for chunk in llm.stream("NPU를 활용한 physical AI 프로젝트 추천해줘."):
    print(chunk.content,end="",flush=True)