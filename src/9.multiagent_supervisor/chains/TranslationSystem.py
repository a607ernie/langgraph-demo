import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import LLMManager


# 定義 LLM 呼叫流程
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
translation_system = """You are a translator converting Taiwanese news articles into English.
Translate the text accurately while maintaining the original tone and style.
Pay special attention to Taiwanese cultural references, idioms, and context.
Ensure that sports team names, player names, and other proper nouns are correctly transliterated or translated as appropriate.
When translating quotes, maintain the speaker's tone and intent."""

translation_prompt = ChatPromptTemplate.from_messages(
    [("system", translation_system), ("human", "Article to translate:\n\n {article}")]
)

translator = translation_prompt | llm

if __name__ == "__main__":
    # Test the Agent
    result = translator.invoke(
        {
            "article": "震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。消息一出，引起球迷熱烈討論。有內部消息指出，小雨此舉可能與新東家開出的天價薪酬有關。究竟是否屬實？本報將持續追蹤報導。敬請球迷朋友們拭目以待，這個夏天的職棒轉會市場肯定會掀起更多驚人巨浪！"
        }
    )

    print(result)