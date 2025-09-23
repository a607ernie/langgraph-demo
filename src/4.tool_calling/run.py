import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from llm import LLMManager

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from utils.graph2mermaid import create_mermaid # for saving mermaid code

# 步驟 1：定義狀態
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def get_taiwan_weather(city: str) -> str:
    """查詢台灣特定城市的天氣狀況。"""
    weather_data = {
        "台北": "晴天，溫度28°C",
        "台中": "多雲，溫度26°C",
        "高雄": "陰天，溫度30°C"
    }
    return f"{city}的天氣：{weather_data.get(city, '暫無資料')}"

# 定義工具
tools = [get_taiwan_weather]
tool_node = ToolNode(tools)

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")


# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

# 步驟 3：添加語言模型節點

def should_continue(state: MessagesState) -> Literal["tools", END]: # type: ignore
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 步驟 4：構建圖
def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", tool_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
    )

    # Any time a tool is called, we return to the agent to decide the next step
    graph_builder.add_edge("tools", "agent")

    graph = graph_builder.compile()

    return graph

# 步驟 6：實現聊天界面
def chat_interface(graph):

    # user_input = "苗栗天氣如何?"
    user_input = "高雄天氣如何?"
    events = graph.stream(
        {"messages": [("user", user_input)]}, # initial state
        config={"recursion_limit": 100}, # we want to limit the recursion depth
        stream_mode="values" 
        # 表示會流式傳輸每個事件的完整狀態值（即整個狀態字典），
        # 而不是只傳輸更新或消息。這允許逐步觀察圖的執行過程。
        # 其他模式包括 "updates"（傳輸節點更新）和 "messages"（傳輸消息）。
        )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    # 運行聊天界面
    # result = get_taiwan_weather.run("台北")
    # print(f"呼叫工具結果{result}")
    graph = build_graph()
    chat_interface(graph)
    
    # create_mermaid(graph)