import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import operator

from llm import LLMManager

from utils.graph2mermaid import create_mermaid # for saving mermaid code
from langgraph.checkpoint.memory import MemorySaver # for saving memory

"""
加入記憶與不加入的差別
在 LangGraph 中，加入 checkpointer（如 MemorySaver）與不加入的主要差別在於狀態的持久化和跨執行會話的記憶。

概念差別：
加入記憶：圖的狀態會被保存到記憶中，允許在多次調用 graph.stream() 之間保持狀態。這對於長對話、多輪交互或需要恢復中斷的應用場景非常有用。
不加入記憶：每次執行 graph.stream() 都是全新的，狀態不會保存，每次都從傳入的初始狀態開始。

加入記憶時，狀態會累積並持久化，適合需要連續對話或狀態管理的應用。
不加入記憶時，每次執行都是獨立的，適合一次性任務或不需要狀態的場景。
"""

# 步驟 1：定義狀態
class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]
    

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# 步驟 3：添加語言模型節點
def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1","count": 1,}
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2","count": 1,}

# 條件邊
def should_continue(state):
    return state["count"] < 3

# 步驟 4：構建圖
def build_graph(memory_enabled=True):
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("Node1", node1)
    graph_builder.add_node("Node2", node2)

    graph_builder.add_edge("Node1", "Node2")
    graph_builder.add_conditional_edges(
        "Node2",
        should_continue,
        {
            True: "Node1",
            False: END
        },
    )

    graph_builder.set_entry_point("Node1")
    
    if not memory_enabled:
        graph = graph_builder.compile() # 步驟 5：編譯圖
        return graph
    
    # Set up memory
    memory = MemorySaver()
    
    graph = graph_builder.compile(checkpointer=memory) # 步驟 5：編譯圖

    return graph

# 步驟 6：實現聊天界面
def chat_interface(graph):

    thread = {
    "configurable": {"thread_id": "1"}}

    print("第一次運行:")
    for event in graph.stream({"count":0, "scratch":"hi"}, thread):
        print(event)

    print("\n第二次運行:")
    for event in graph.stream({"count":0, "scratch":"hi"}, thread):  # 注意：這裡傳入 count:0，但記憶會恢復
        print(event)

    print("\n第三次運行 (thread_id=2):")
    thread2 = {"configurable": {"thread_id": "2"}}
    for event in graph.stream({"count":0, "scratch":"hi"}, thread2):
        print(event)

if __name__ == "__main__":
    # 運行聊天界面
    memory_enabled = True  # 設置為 True 或 False 以啟用或禁用記憶
    graph = build_graph(memory_enabled=memory_enabled)
    # chat_interface(graph)

    create_mermaid(graph)