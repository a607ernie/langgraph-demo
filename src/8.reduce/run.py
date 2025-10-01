from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# 定義 State
class State(TypedDict):
    # 使用 Annotated + add_messages => LangGraph 會自動 "reduce" 合併
    messages: Annotated[list[str], add_messages]


# 節點 A：輸出一則訊息
def node_a(state: State):
    return {"messages": ["Hello"]}


# 節點 B：再輸出一則訊息
def node_b(state: State):
    return {"messages": ["World"]}


# 建立 Graph
builder = StateGraph(State)

builder.add_node("A", node_a)
builder.add_node("B", node_b)

builder.set_entry_point("A")
builder.add_edge("A", "B")
builder.add_edge("B", END)

graph = builder.compile()

# 執行 Graph
final_state = graph.invoke({"messages": []})

print(final_state)
