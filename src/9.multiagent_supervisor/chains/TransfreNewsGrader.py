import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from llm import LLMManager

# 定義輸出格式（介面）
class CheerleaderNewsGrader(BaseModel):
    """Binary score for relevance check on Taiwanese professional baseball cheerleader news."""

    binary_score: str = Field(
        description="The article is about Taiwanese professional baseball cheerleaders, 'yes' or 'no'"
    )

# 定義 LLM 呼叫流程
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
structured_llm_grader = llm.with_structured_output(CheerleaderNewsGrader)

system = """You are a grader assessing whether a news article concerns Taiwanese professional baseball cheerleaders.
    Check if the article explicitly mentions:
    1. Cheerleader transfers between CPBL (Chinese Professional Baseball League) teams
    2. New cheerleader recruitment or retirement
    3. Special performances or events featuring the cheerleaders
    4. Controversies or notable incidents involving cheerleaders
    5. Changes in cheerleading teams' leadership or management
    Provide a binary score 'yes' or 'no' to indicate whether the news is about Taiwanese professional baseball cheerleaders."""

grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "News Article:\n\n {article}")]
)

evaluator = grade_prompt | structured_llm_grader

if __name__ == "__main__":
    # 測試 Agent 運作狀況
    result = evaluator.invoke(
        {"震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊"}
    )

    print(result)