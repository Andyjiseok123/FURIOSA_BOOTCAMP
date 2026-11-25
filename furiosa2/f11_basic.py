"""
furiosa-llm serve furiosa-ai/Qwen3-8B-FP8 \
    --devices npu:0 \
    --host 127.0.0.1 \
    --port 8000 \
    --enable-overlap-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
"""

from kingkeras import get_llm

llm = get_llm()

prompt = 'NPU와 GPU의 차이를 설명해줘.'

response = llm.invoke(prompt)

print(response.content)

