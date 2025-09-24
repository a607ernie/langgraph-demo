import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import LLMManager

# 定義輸出格式（介面）
class TaiwanArticlePostabilityGrader(BaseModel):
    """Binary scores for postability check, word count, sensationalism, and language verification of a Taiwanese news article."""

    can_be_posted: str = Field(
        description="The article is ready to be posted, 'yes' or 'no'"
    )
    meets_word_count: str = Field(
        description="The article has at least 300 characters, 'yes' or 'no'"
    )
    is_sensationalistic: str = Field(
        description="The article is written in a sensationalistic style, 'yes' or 'no'"
    )
    is_language_traditional_chinese: str = Field(
        description="The language of the article is Traditional Chinese, 'yes' or 'no'"
    )


# 定義 LLM 呼叫流程
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
structured_llm_postability_grader = llm.with_structured_output(TaiwanArticlePostabilityGrader)

postability_system = """You are a grader assessing whether a Taiwanese news article is ready to be posted, if it meets the minimum character count of 300 characters, is written in a sensationalistic style, and if it is in Traditional Chinese. \n
    Evaluate the article for grammatical errors, completeness, appropriateness for publication, and EXAGGERATED sensationalism. \n
    Also, confirm if the language used in the article is Traditional Chinese and it meets the character count requirement. \n
    Provide four binary scores: one to indicate if the article can be posted ('yes' or 'no'), one for adequate character count ('yes' or 'no'), one for sensationalistic writing ('yes' or 'no'), and another if the language is Traditional Chinese ('yes' or 'no').\n
    Pay attention to Taiwan-specific terms, idioms, and writing styles."""

postability_grade_prompt = ChatPromptTemplate.from_messages(
    [("system", postability_system), ("human", "News Article:\n\n {article}")]
)

news_chef = postability_grade_prompt | structured_llm_postability_grader

if __name__ == "__main__":
    # 測試 Agent 運作狀況
    result = news_chef.invoke(
        {
            "article": "震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。消息一出，引起球迷熱烈討論。有內部消息指出，小雨此舉可能與新東家開出的天價薪酬有關。究竟是否屬實？本報將持續追蹤報導。敬請球迷朋友們拭目以待，這個夏天的職棒轉會市場肯定會掀起更多驚人巨浪！"
        }
    )
    print(result)