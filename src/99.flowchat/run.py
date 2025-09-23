import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from utils.graph2mermaid import create_mermaid # for saving mermaid code

# 步驟 1：定義狀態
class MyState(TypedDict):  # from typing import TypedDict
    i: int
    j: int
    k: int
    valid: str
    reask: bool = False

# Functions on **nodes**
def parse(state: MyState):
    print(f"parse: {state['i'], state['j'], state['k']}")
    if state.get('reask', False):
        print("Re-asking for input due to previous invalid input.")
        user_input = input("Enter three integers separated by spaces: ")
        nums = [int(x) for x in user_input.split()]
        i = nums[0] if len(nums) > 0 else state['i']
        j = nums[1] if len(nums) > 1 else state['j']
        k = nums[2] if len(nums) > 2 else state['k']
    else:
        i, j, k = state['i'], state['j'], state['k']
    return {"i": i, "j": j, "k": k}

def validate(state: MyState):
    i, j, k = state["i"], state["j"], state["k"]
    
    print(f"validate: {i}, {j}, {k}")
    if i is None or j is None or k is None:
        return {"i": i, "j": j, "k": k, "valid": "INVALID", "reask": True}
    
    if i+j+k < 200:
        return {"i": i, "j": j, "k": k, "valid": "TOO SMALL", "reask": True}

    return {"i": i, "j": j, "k": k, "valid": "OK", "reask": False}

# def validate_i(state: MyState):
#     i = state["i"]
#     print(f"validate_i: {i}")
#     if i > 100:
#         return {"i": i, "valid": "TOO BIG:I"}
#     return {"i": i, "valid": "OK"}

# Conditional **edge** function
def is_small_enough(state: MyState):
    if state['valid'] == "OK":
        return END
    else:
        return "parse"

# 步驟 4：構建圖
def build_graph():
    workflow = StateGraph(MyState)

    workflow.add_node("parse", parse)
    workflow.add_node("validate", validate)

    workflow.set_entry_point("parse")

    workflow.add_edge("parse", "validate")
    workflow.add_conditional_edges(
        source="validate", path=is_small_enough
    )
    graph = workflow.compile()
    return graph



if __name__ == "__main__":
    # 運行聊天界面
    
    graph = build_graph()

    # create_mermaid(graph)

    user_input = {"i": 0, "j": 0, "k": 0}
    r = graph.stream(user_input)
    for item in r:
        print(item)
    