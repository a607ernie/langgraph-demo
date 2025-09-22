import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import operator
# Annotated 用於為類型添加額外元數據，例如在這裡用於指定 list 的處理方式
# TypedDict 是一種特殊的字典類型，用於定義具有固定鍵
# Sequence 用於表示一個有序的元素集合，例如 list 或 tuple
from typing import TypedDict, Annotated, Sequence 
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.prompts import ChatPromptTemplate
from llm import LLMManager
from langchain_core.messages import AIMessage
from PIL import Image
from io import BytesIO

# 步驟 1：定義狀態
class AllState(TypedDict):
    messages: Annotated[list, operator.add]

# 步驟 2：定義語言模型
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# add chain
def extract_city_name(messages: list) -> str:
    # 定義一個鏈來提取城市名稱
    user_query = messages
    prompt_str = """
    系統會給出一個問題，要求你從中提取城市名稱。
    除了城市名稱外，不要回答任何其他問題；如果找不到城市名稱，也不要回答任何內容。
    如果城市名稱存在，則僅回答該城市名稱；如果不存在該城市名稱，則回覆「no_response」。

    問題如下：
    {user_query}
    """
    prompt = ChatPromptTemplate.from_template(prompt_str)

    chain = prompt | llm

    response = chain.invoke({"user_query": user_query})
    return response

def create_response_chain(user_query: str, information: str):
    # 定義一個鏈來根據用戶查詢和天氣資訊生成
    response_prompt_str = """
    您提供了一條天氣訊息，您需要根據該資訊回覆用戶的查詢。
    用戶查詢如下：
    ---
    {user_query}
    ---

    資訊如下：
    ---
    {information}
    ---

    """
    response_prompt = ChatPromptTemplate.from_template(response_prompt_str) # 使用模板創建提示

    response_chain = response_prompt | llm # 創建鏈
    response = response_chain.invoke({'user_query': user_query, 'information': information}) # 調用鏈
    return response

# 步驟 3：添加節點
def get_taiwan_weather(city: str) -> str:
    """查詢台灣特定城市的天氣狀況。"""
    weather_data = {
        "台北": "晴天，溫度28°C",
        "台中": "多雲，溫度26°C",
        "高雄": "陰天，溫度30°C"
    }
    return f"{city}的天氣：{weather_data.get(city, '暫無資料')}"

def call_model(state: AllState):
    messages = state["messages"]
    
    response = extract_city_name(messages)
    return {"messages": [response]}

def weather_tool(state: AllState):
  context = state["messages"]
  city_name = context[1].content
  data = get_taiwan_weather(city_name)
  return {"messages": [AIMessage(content=data)]}

def responder(state: AllState):
    messages = state["messages"]
    response = create_response_chain(
        user_query=messages[0],
        information=messages[-1].content
    )
    return {"messages": [response]}

# 添加分支節點
def query_classify(state: AllState):
  messages = state["messages"]
  ctx = messages[1].content
  if ctx == "no_response":
    return "end"
  else:
    return "continue"


# 步驟 4：構建圖
def build_graph():
    graph_builder = StateGraph(AllState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("weather", weather_tool)
    graph_builder.add_node("responder", responder)
    
    graph_builder.add_conditional_edges('agent', query_classify, {
        "continue": "weather",
        "end": END
    })

    graph_builder.add_edge('weather', 'responder')
    graph_builder.add_edge('responder', END)

    graph_builder.set_entry_point("agent")

    graph = graph_builder.compile()

    return graph

# 生成 Mermaid 語法
def generate_mermaid(graph):
    """從 LangGraph 圖中生成 Mermaid 語法"""
    return graph.get_graph().draw_mermaid()

def create_mermaid():
    graph = build_graph()
    # 生成 PNG 數據
    png_data = graph.get_graph().draw_mermaid_png()
    # 轉換到 JPG
    image = Image.open(BytesIO(png_data))
    image.save("src/weather_search/graph.jpg", "JPEG")
    print("Graph 的 JPG 圖片已保存為 graph.jpg")

# 步驟 5：實現聊天界面
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
    # create_mermaid()
