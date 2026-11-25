from kingkeras import get_llm
from langchain_core.tools import tool

@tool
def add(aaa:int, bbb:int) -> int:
    """두 정수를 더합니다."""
    return aaa+ bbb

print(add.name)
print(add.description)
print(add.args)