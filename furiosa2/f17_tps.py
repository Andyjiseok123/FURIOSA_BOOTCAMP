from kingkeras import get_llm
import time
llm = get_llm(max_token=500)

prompt="NPU의 특징을 자세히 설명해줘."

start = time.perf_counter()

response = llm.invoke(prompt)

elapsed = time.perf_counter()-start

tokens = response.usage_metadata["output_tokens"]

tps = tokens/elapsed

print("출력토큰 : ", tokens)
print("전체시간 : ", elapsed,"초")
print("TPS : ",round(tps,2))


"""
출력토큰 :  500
전체시간 :  5.504987861029804 초
TPS :  90.83
"""