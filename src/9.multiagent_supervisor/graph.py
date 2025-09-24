import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated, List, Tuple, Union, Literal,TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from utils.graph2mermaid import create_mermaid # for saving mermaid code
from langchain_core.prompts import ChatPromptTemplate

# import the chains
from chains.ExpansionSystem import expander
from chains.ArticlePostabilityGrader import news_chef
from chains.TransfreNewsGrader import evaluator
from chains.TranslationSystem import translator

class AgentState(TypedDict):
    article_state: str


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

def build_graph() -> StateGraph:
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

def test_case_1(graph: StateGraph):
    test_case_1 = {
        "article_state": "知名啦啦隊女神小雨宣布退出Lamigo桃猿，轉戰統一獅啦啦隊。"
    }
    result_1 = graph.invoke(test_case_1)
    print(result_1)

def test_case_2(graph: StateGraph):
    """
    evaluator 應該判定這不是啦啦隊或棒球相關新聞。
    工作流程應該直接結束，不進行後續處理。
    """
    test_case_2 = {
        "article_state": "台北市今日發布最新空氣品質報告，PM2.5指數持續攀升。"
    }
    result_2 = graph.invoke(test_case_2)
    print(result_2)

def test_case_3(graph: StateGraph):
    """
    evaluator 應該判定這是相關的棒球新聞。
    news_chef 會發現文章是英文，需要翻譯。
    translator 會將文章翻譯成繁體中文。
    翻譯後的文章會再次通過 news_chef 評估。
    如果文章仍然較短，可能會被送到 expander 進行擴展。 最終，如果符合發布標準，文章會被送到 publisher。
    """
    test_case_3 = {
        "article_state": "CPBL star outfielder Wang Po-Jung considering a return to NPB after successful stint with Lamigo Monkeys."
    }
    result_3 = graph.invoke(test_case_3)
    print(result_3)

if __name__ == "__main__":
    graph = build_graph()
    # create_mermaid(graph)

    # test_case_1(graph)
    # test_case_2(graph)
    test_case_3(graph)