import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated, List, Tuple, TypedDict, Literal

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from llm import LLMManager
from langchain_core.tools import tool

from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode

from utils.graph2mermaid import create_mermaid # for saving mermaid code
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

@tool
def search_taiwan_info(query: str):
    """搜尋台灣相關資訊。"""
    # 這是一個示例實現
    return [
        "台北今天陽光明媚，適合出遊。"
    ]


tools = [search_taiwan_info]
tool_node = ToolNode(tools)


# 定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
bound_model = llm.bind_tools(tools)

# 步驟 3：添加語言模型節點
"""定義節點與流程控制函數"""
def should_continue(state: MessagesState) -> Literal["action", "__end__"]:
    """決定下一個執行的節點。"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "__end__"
    return "action"

def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    return {"messages": response}

# 步驟 4：構建圖
def build_graph():
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("action", "agent")

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph

# 步驟 6：實現聊天界面
def chat_interface(graph):

    config = {"configurable": {"thread_id": "taiwan_chat"}}
    print("\n****** 第一次對話 ******\n")
    input_message = HumanMessage(content="你好！我是小明")
    for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()
    
    print("\n****** 第二次對話 ******\n")
    input_message = HumanMessage(content="我的名字是什麼？")
    for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


    anonymous_config = {"configurable": {"thread_id": "anonymous_chat"}}
    print("\n****** 第三次對話 (匿名) ******\n")
    input_message = HumanMessage(content="我的名字是什麼？")
    for event in graph.stream({"messages": [input_message]}, anonymous_config, stream_mode="values"):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    # 運行聊天界面
    graph = build_graph()
    chat_interface(graph)
    # create_mermaid(graph)