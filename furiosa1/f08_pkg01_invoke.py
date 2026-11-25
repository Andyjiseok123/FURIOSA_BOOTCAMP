from furiosa1.a_kingkeras import get_llm

llm = get_llm(
    max_tokens=1000,
    temperature=0.9,
    top_p=1.0,
)


response = llm.invoke("NPU를 활용한 physical AI 프로젝트 추천해줘.")

print(response.content)