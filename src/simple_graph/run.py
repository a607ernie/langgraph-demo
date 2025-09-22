import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from llm import LLMManager

# 步驟 1：定義狀態
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# 步驟 3：添加語言模型節點
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 步驟 4：構建圖
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")

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
                print("AI 助理:", value["messages"][-1].content)

if __name__ == "__main__":
    # 運行聊天界面
    chat_interface()