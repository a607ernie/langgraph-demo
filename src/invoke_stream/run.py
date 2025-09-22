import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated  # Annotated 用於為類型添加額外元數據，例如在這裡用於指定 list 的處理方式
from typing_extensions import TypedDict  # TypedDict 是一種特殊的字典類型，用於定義具有固定鍵和類型的字典結構，與普通 dict 不同，它提供類型檢查和更好的 IDE 支持
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages  # add_messages 是一個函數，用於將新消息添加到現有的消息列表中，常用於處理對話歷史

from llm import LLMManager

# 步驟 1：定義狀態
class State(TypedDict):
    messages: Annotated[list, add_messages]  # messages 是一個列表，使用 add_messages 來處理消息的添加邏輯

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# 步驟 3：添加語言模型節點
def preprocess(state: State):
    # 前處理節點：添加一個系統消息
    system_message = ("system", "你是一個友好的 AI 助手，請用簡潔的方式回應用戶。")
    return {"messages": [system_message] + state["messages"]}

def chatbot(state: State):
    # 這個函數代表圖中的一個節點，它接收當前狀態，調用 LLM 生成回應，並返回更新後的狀態
    # 這裡的動作是將用戶的消息傳遞給 LLM，並將回應添加到 messages 中
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph():
    # 步驟 4：構建圖
    graph_builder = StateGraph(State)
    graph_builder.add_node("preprocess", preprocess)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.set_entry_point("preprocess")
    graph_builder.add_edge("preprocess", "chatbot")
    graph_builder.set_finish_point("chatbot")


    # 步驟 5：編譯圖
    graph = graph_builder.compile()

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
        # graph.stream 用於逐步執行圖，並返回每個步驟的事件流，允許實時處理中間結果
        # 與 graph.invoke 不同，invoke 是同步執行整個圖並返回最終結果，而 stream 允許流式處理
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("AI 助理:", value["messages"][-1].content)


# 演示 graph.invoke 與 graph.stream 的差異
def demonstrate_invoke_vs_stream():
    graph = build_graph()
    user_input = "你好，請介紹一下自己。"

    print("=== graph.invoke (一次性執行，直接獲取最終結果) ===")
    # invoke: 直接得到最終結果，跳過中間步驟
    final_result = graph.invoke({"messages": [("user", user_input)]})
    print("最終回應:", final_result["messages"][-1].content)

    print("\n=== graph.stream (逐步執行，可見中間步驟) ===")
    # stream: 逐步得到結果，顯示每個節點的輸出
    print("逐步回應:")
    step = 1
    for event in graph.stream({"messages": [("user", user_input)]}):
        print(f"步驟 {step}:")
        for node_name, value in event.items():
            print(f"  節點 '{node_name}' 輸出: {value['messages'][-1].content}")
        step += 1


if __name__ == "__main__":
    # 運行聊天界面
    demonstrate_invoke_vs_stream()
    # chat_interface()