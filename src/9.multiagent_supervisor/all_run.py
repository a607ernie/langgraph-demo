import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated, List, Tuple, Union, Literal,TypedDict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from llm import LLMManager

from utils.graph2mermaid import create_mermaid # for saving mermaid code
from datetime import datetime
import pytz
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

current_time = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %Z")


# 步驟 1：定義狀態
class AgentState(TypedDict):
    article_state: str  # 文章狀態

"""定義資料模型"""
# 建置 TransfreNewsGrader，啦啦隊相關新聞評估
class CheerleaderNewsGrader(BaseModel):
    """Binary score for relevance check on Taiwanese professional baseball cheerleader news."""

    binary_score: str = Field(
        description="The article is about Taiwanese professional baseball cheerleaders, 'yes' or 'no'"
    )

# 建置 ArticlePostabilityGrader，文章可發佈性評估
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


"""定義語言模型 & structured output"""
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")
structured_llm_grader = llm.with_structured_output(CheerleaderNewsGrader)
structured_llm_postability_grader = llm.with_structured_output(TaiwanArticlePostabilityGrader)


"""建立提示詞"""
system = """You are a grader assessing whether a news article concerns Taiwanese professional baseball cheerleaders.
    Check if the article explicitly mentions:
    1. Cheerleader transfers between CPBL (Chinese Professional Baseball League) teams
    2. New cheerleader recruitment or retirement
    3. Special performances or events featuring the cheerleaders
    4. Controversies or notable incidents involving cheerleaders
    5. Changes in cheerleading teams' leadership or management
    Provide a binary score 'yes' or 'no' to indicate whether the news is about Taiwanese professional baseball cheerleaders."""

postability_system = """You are a grader assessing whether a Taiwanese news article is ready to be posted, if it meets the minimum character count of 300 characters, is written in a sensationalistic style, and if it is in Traditional Chinese. \n
    Evaluate the article for grammatical errors, completeness, appropriateness for publication, and EXAGGERATED sensationalism. \n
    Also, confirm if the language used in the article is Traditional Chinese and it meets the character count requirement. \n
    Provide four binary scores: one to indicate if the article can be posted ('yes' or 'no'), one for adequate character count ('yes' or 'no'), one for sensationalistic writing ('yes' or 'no'), and another if the language is Traditional Chinese ('yes' or 'no').\n
    Pay attention to Taiwan-specific terms, idioms, and writing styles."""

translation_system = """You are a translator converting Taiwanese news articles into English.
Translate the text accurately while maintaining the original tone and style.
Pay special attention to Taiwanese cultural references, idioms, and context.
Ensure that sports team names, player names, and other proper nouns are correctly transliterated or translated as appropriate.
When translating quotes, maintain the speaker's tone and intent."""

expansion_system = """你是一位專業的台灣新聞記者，負責將給定的簡短新聞擴展至至少 300 字。在擴展過程中，請注意以下幾點：

1. 保持原文的主題和tone，同時增加相關的背景資訊和細節。
2. 使用台灣讀者熟悉的表達方式和用語。
3. 適當加入一些專家或相關人士的假想評論，以增加新聞的深度。
4. 考慮新聞事件可能對台灣社會或特定群體的影響。
5. 在適當的地方加入一些台灣特有的文化元素或本地化的例子。
6. 確保擴展後的文章仍然保持客觀性和新聞專業性。
7. 使用繁體中文撰寫。"""

"""建立提示模板"""
grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "News Article:\n\n {article}")]
)

postability_grade_prompt = ChatPromptTemplate.from_messages(
    [("system", postability_system), ("human", "News Article:\n\n {article}")]
)

translation_prompt = ChatPromptTemplate.from_messages(
    [("system", translation_system), ("human", "Article to translate:\n\n {article}")]
)

expansion_prompt = ChatPromptTemplate.from_messages(
    [("system", expansion_system), ("human", "原始新聞內容：\n\n {article}")]
)

"""建立chain"""
evaluator = grade_prompt | structured_llm_grader # 建置 TransfreNewsGrader，啦啦隊相關新聞評估
news_chef = postability_grade_prompt | structured_llm_postability_grader  # 建置 ArticlePostabilityGrader，文章可發佈性評估
translator = translation_prompt | llm # 建置 translation_system，文章翻譯
expander = expansion_prompt | llm # 建置 expansion_system，文章擴展

# 步驟 3：添加語言模型節點
### 呼叫 Agnet 工作以及顯示節點狀態用
def get_transfer_news_grade(state: AgentState) -> AgentState:
    print(f"get_transfer_news_grade: Current state: {state}")
    print("Evaluator: Reading article but doing nothing to change it...")
    return state
def evaluate_article(state: AgentState) -> AgentState:
    print(f"evaluate_article: Current state: {state}")
    print("News : Reading article but doing nothing to change it...")
    return state
def translate_article(state: AgentState) -> AgentState:
    print(f"translate_article: Current state: {state}")
    article = state["article_state"]
    result = translator.invoke({"article": article})
    state["article_state"] = result.content
    return state
def expand_article(state: AgentState) -> AgentState:
    print(f"expand_article: Current state: {state}")
    article = state["article_state"]
    result = expander.invoke({"article": article})
    state["article_state"] = result.content
    return state
def publisher(state: AgentState) -> AgentState:
    print(f"publisher: Current state: {state}")
    print("FINAL_STATE in publisher:", state)
    return state

## 提供路由使用
def evaluator_router(state: AgentState) -> Literal["news_chef", "not_relevant"]:
    article = state["article_state"]
    evaluator = grade_prompt | structured_llm_grader
    result = evaluator.invoke({"article": article})
    print(f"evaluator_router: Current state: {state}")
    print("Evaluator result: ", result)
    if result.binary_score == "yes":
        return "news_chef"
    else:
        return "not_relevant"
    

def news_chef_router(
    state: AgentState,
) -> Literal["translator", "publisher", "expander"]:
    article = state["article_state"]
    result = news_chef.invoke({"article": article})
    print(f"news_chef_router: Current state: {state}")
    print("News chef result: ", result)
    if result.can_be_posted == "yes":
        return "publisher"
    elif result.is_language_traditional_chinese == "yes":
        if result.meets_word_count == "no" or result.is_sensationalistic == "no":
            return "expander"
    return "translator"

# 步驟 4：構建圖
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("evaluator", get_transfer_news_grade)
    workflow.add_node("news_chef", evaluate_article)
    workflow.add_node("translator", translate_article)
    workflow.add_node("expander", expand_article)
    workflow.add_node("publisher", publisher)

    workflow.set_entry_point("evaluator")

    workflow.add_conditional_edges(
        "evaluator", evaluator_router, {"news_chef": "news_chef", "not_relevant": END}
    )
    workflow.add_conditional_edges(
        "news_chef",
        news_chef_router,
        {"translator": "translator", "publisher": "publisher", "expander": "expander"},
    )
    workflow.add_edge("translator", "news_chef")
    workflow.add_edge("expander", "news_chef")
    workflow.add_edge("publisher", END)

    graph = workflow.compile()

    return graph

# 步驟 6：實現聊天界面
def chat_interface():

    print("歡迎使用 AI 助理！輸入 'quit', 'exit' 或 'q' 來結束對話。")
    while True:
        user_input = input("使用者: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("掰啦!")
            break

        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("AI 助理:", value["messages"][-1].content)

def simple_test():
    """測試"""
    result = evaluator.invoke(
        {"震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊"}
    )

    print(result)

    result = news_chef.invoke(
        {
            "article": "震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。消息一出，引起球迷熱烈討論。有內部消息指出，小雨此舉可能與新東家開出的天價薪酬有關。究竟是否屬實？本報將持續追蹤報導。敬請球迷朋友們拭目以待，這個夏天的職棒轉會市場肯定會掀起更多驚人巨浪！"
        }
    )
    print(result)

    result = translator.invoke(
        {
            "article": "震撼彈！知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。消息一出，引起球迷熱烈討論。有內部消息指出，小雨此舉可能與新東家開出的天價薪酬有關。究竟是否屬實？本報將持續追蹤報導。敬請球迷朋友們拭目以待，這個夏天的職棒轉會市場肯定會掀起更多驚人巨浪！"
        }
    )

    print(result)

    article_content = "知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。"
    result = expander.invoke({"article": article_content})

    print(result)

if __name__ == "__main__":
    # 運行聊天界面
    graph = build_graph()

    test_case_1 = {
        "article_state": "知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。"
    }
    result_1 = graph.invoke(test_case_1)
    print(result_1)
    
    # chat_interface(graph)
    # create_mermaid(graph)