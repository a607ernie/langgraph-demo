import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated,TypedDict, List, Tuple, Union, Literal

# langchain,langgraph 相關
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import filter_messages, AIMessage, HumanMessage, SystemMessage, trim_messages


# utils & llm 相關
from llm import LLMManager
from utils.graph2mermaid import create_mermaid # for saving mermaid code


messages = [
    SystemMessage("你是一個優秀的助理。"),
    HumanMessage("你叫什麼名字？", id="q1", name="台灣使用者"),
    AIMessage("我叫小智。", id="a1", name="AI助理"),
    HumanMessage("你最喜歡的台灣小吃是什麼？", id="q2"),
    AIMessage("我最喜歡的是臺灣的珍珠奶茶！", id="a2"),
]

filtered_msgs = filter_messages(
    messages,
    include_names=("台灣使用者", "AI助理"),
    include_types=("system",),
    exclude_ids=("a1",),
)

print(filtered_msgs)


messages = [
    SystemMessage(content="你是一個了解台灣文化的助理"),
    HumanMessage(content="你好！我是小明"),
    AIMessage(content="你好小明！很高興認識你。"),
    HumanMessage(content="我最喜歡吃臺灣的牛肉麵"),
    AIMessage(content="牛肉麵確實是台灣很受歡迎的美食！"),
    HumanMessage(content="謝謝你的回答"),
    AIMessage(content="不客氣，很高興能幫到你！"),
    HumanMessage(content="你喜歡台灣嗎？"),
    AIMessage(content="當然！台灣有豐富的文化和美食，我很喜歡。"),
]

llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

# 自定義 token 計數函數，因為模型不支援內建計數
def count_tokens(messages):
    """近似 token 計數：假設每個字符約為 1 個 token（適用於中文）"""
    total_chars = sum(len(str(msg.content)) for msg in messages)
    return total_chars

trimmer = trim_messages(
    max_tokens=50,
    strategy="last",
    token_counter=count_tokens,  # 使用自定義計數函數
    include_system=True,  # 保留初始的系統訊息
    allow_partial=False,  # 不允許分割訊息內容
    start_on="human",  # 確保第一條訊息（不包括系統訊息）始終是特定類型
)

# 先修剪消息並印出
trimmed_messages = trimmer.invoke(messages)
print("修剪後的消息：")
print(trimmed_messages)
print(f"修剪後的 token 數量：{count_tokens(trimmed_messages)}")

chain = trimmer | llm
result = chain.invoke(messages)
print("最終結果：")
print(result)