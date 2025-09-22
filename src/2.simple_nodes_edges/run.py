import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END # 入口點(Entry Point)和終點(End Point)

from llm import LLMManager

# 步驟 1：定義狀態
class AllState(TypedDict):
    messages: Annotated[list, operator.add] # messages 是一個列表，使用 operator.add 來處理消息的添加邏輯

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# 步驟 3：添加語言模型節點
def function1(state):
    last_message = state["messages"][-1][1]
    new_content = last_message + " Function1處理完畢"
    return {"messages": state["messages"] + [("assistant", new_content)]}

def function2(state):
    last_message = state["messages"][-1][1]
    new_content = last_message + " Function2處理完畢"
    return {"messages": state["messages"] + [("assistant", new_content)]}

# 定義條件邊
def where_to_go(state):
  # Your Logic here
  if state['Condition']:
    return "end"
  else:
    return "continue"

# 步驟 4：構建圖
def build_graph():
    graph_builder = StateGraph(AllState)
    
    graph_builder.add_node("node1", function1)
    graph_builder.add_node("node2", function2)

    graph_builder.add_edge(START, "node1")
    graph_builder.add_edge("node1", "node2")
    graph_builder.add_edge("node2", END)


    # agent node is connected with 2 nodes END and node2
    # graph.add_conditional_edges('agent',where_to_go,{
    #     "end": END,
    #     "continue": "node2"
    # })

    graph = graph_builder.compile() # 步驟 5：編譯圖

    return graph

# 步驟 6：實現聊天界面
def chat_interface():
    graph = build_graph()

    print("歡迎使用 AI 助理！輸入 'quit', 'exit' 或 'q' 來結束對話。")
    while True:
        user_input = input("使用者: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("掰啦!")
            break

        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("AI 助理:", value["messages"][-1][1])

if __name__ == "__main__":
    # 運行聊天界面
    chat_interface()