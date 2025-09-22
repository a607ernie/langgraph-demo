import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from io import BytesIO
from PIL import Image
from llm import LLMManager

# 步驟 1：定義狀態
class MyState(TypedDict):  # from typing import TypedDict
    i: int
    j: int

# Functions on **nodes**
def fn1(state: MyState):
    print(f"Enter fn1: {state['i']}")
    return {"i": 1}

def fn2(state: MyState):
    i = state["i"]
    print(f"fn2: {i}")
    return {"i": i+1}

# Conditional **edge** function
def is_big_enough(state: MyState):
    if state["i"] > 10:
        return END
    else:
        return "n2"

# 步驟 4：構建圖
def build_graph():
    workflow = StateGraph(MyState)

    workflow.add_node("n1", fn1)
    workflow.add_node("n2", fn2)

    workflow.set_entry_point("n1")

    workflow.add_edge("n1", "n2")
    workflow.add_conditional_edges(
        source="n2", path=is_big_enough
    )
    graph = workflow.compile()
    return graph

def create_mermaid():
    graph = build_graph()
    # 生成 PNG 數據
    png_data = graph.get_graph().draw_mermaid_png()
    # 轉換到 JPG
    image = Image.open(BytesIO(png_data))
    image.save(os.path.join(os.path.dirname(__file__), "graph.jpg"), "JPEG")
    print("Graph 的 JPG 圖片已保存為 graph.jpg")

if __name__ == "__main__":
    # 運行聊天界面
    # create_mermaid()
    graph = build_graph()
    r = graph.invoke({"i": 1, "j": 123})
    print(r)