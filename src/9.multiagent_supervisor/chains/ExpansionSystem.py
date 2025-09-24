import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import LLMManager


# 定義 LLM 呼叫流程
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
expansion_system = """你是一位專業的台灣新聞記者，負責將給定的簡短新聞擴展至至少 300 字。在擴展過程中，請注意以下幾點：

1. 保持原文的主題和tone，同時增加相關的背景資訊和細節。
2. 使用台灣讀者熟悉的表達方式和用語。
3. 適當加入一些專家或相關人士的假想評論，以增加新聞的深度。
4. 考慮新聞事件可能對台灣社會或特定群體的影響。
5. 在適當的地方加入一些台灣特有的文化元素或本地化的例子。
6. 確保擴展後的文章仍然保持客觀性和新聞專業性。
7. 使用繁體中文撰寫。"""

expansion_prompt = ChatPromptTemplate.from_messages(
    [("system", expansion_system), ("human", "原始新聞內容：\n\n {article}")]
)

expander = expansion_prompt | llm

if __name__ == "__main__":
    # 測試 Agent 運作狀況
    article_content = "知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。"
    result = expander.invoke({"article": article_content})

    print(result)
     