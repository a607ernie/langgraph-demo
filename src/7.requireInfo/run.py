import sys
import os
import inspect

from pydantic import BaseModel, Field
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Annotated, Optional, TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph import add_messages
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from llm import LLMManager
from typing import Literal, Any, Dict, List
from utils.graph2mermaid import create_mermaid # for saving mermaid code

## 定義使用者資訊
class RequiredInformation(BaseModel):
    provided_full_name: Optional[str] = Field(
        default=None,
        description="the provided full name of the user"
    )
    provided_mobile: Optional[str] = Field(
        default=None,
        description="the provided mobile number of the user"
    )
    provided_id_4_digits: Optional[int] = Field(
        default=None,
        description="the provided user last 4 digits of id card"
    )

"""定義 Graph 中狀態管理"""
class AssistantGraphState(TypedDict):
    user_question: str
    required_information: RequiredInformation
    messages: Annotated[list, add_messages]

"""定義語言模型"""
llm_manager = LLMManager()
llm = llm_manager.get_llm("chat")

"""定義系統提示"""
system = f"""你是 AI 客服助理。你的任務是收集必要的用戶資訊。請遵循以下原則:

1. 保持禮貌和專業,使用適當的敬語。
2. 如果用戶詢問的資訊不完整,請適當地要求補充。
3. 在收集用戶資訊時,請確保隱私和安全。
4. 如果無法回答某個問題,請誠實地表示,並提供其他可能的幫助方式。


需要收集的資訊包括：

{inspect.getsource(RequiredInformation)}

請確保每項資訊都符合要求後再進行下一項。

DO NOT FILL IN THE USERS INFORMATION, YOU NEED TO COLLECT IT.

請根據用戶的問題和已提供的資訊,給出適當的回應和指引。"""

collect_info_system = f"""你是 AI 客服助理。你的任務是收集必要的用戶資訊。請遵循以下原則:

1. 保持禮貌和專業,使用適當的敬語。
2. 如果用戶詢問的資訊不完整,請適當地要求補充。
3. 在收集用戶資訊時,請確保隱私和安全。
4. 如果無法回答某個問題,請誠實地表示,並提供其他可能的幫助方式。


需要收集的資訊包括：

{inspect.getsource(RequiredInformation)}

請確保每項資訊都符合要求後再進行下一項。

DO NOT FILL IN THE USERS INFORMATION, YOU NEED TO COLLECT IT.

請根據用戶的問題和已提供的資訊,給出適當的回應和指引。
"""

# 定義回應建構器的系統提示
response_builder_system = """
你是台灣高鐵的AI客服助理。你的任務是總結對話內容，並提供一個清晰、專業的回應給用戶。請遵循以下原則：

1. 總結已收集的用戶資訊（如果有的話）。
2. 簡要回顧對話中討論的主要問題或請求。
3. 提供任何相關的後續步驟或建議。
4. 使用禮貌和專業的語氣。
5. 如果有任何未完成的事項，請提醒用戶。

請確保你的回應簡潔但全面，並符合高鐵客服的專業標準。
"""

"""定義提示模板"""
# 創建助理提示模板
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: {user_question}\n"
            "Chat history: {messages}\n"
            "\n\n What the user have provided so far {provided_required_information} \n\n"
        ),
    ]
)
# 創建助理提示模板
collect_info_prompt = ChatPromptTemplate.from_messages([
    ("system", collect_info_system),
    (
        "human",
        "User question: {user_question}\n"
        "Chat history: {messages}\n"
        "\n\n What the user have provided so far {provided_required_information} \n\n"
    ),
])

response_builder_prompt = ChatPromptTemplate.from_messages([
    ("system", response_builder_system),
    ("human", "用戶資訊：{user_info}\n對話歷史：{chat_history}\n請提供一個總結性的回應。")
])

"""定義流程控制函數"""
def provided_all_details(state: AssistantGraphState) -> Literal["info all collected", "not fulfill"]:
    if "required_information" not in state:
        return "not fulfill"
    provided_information: RequiredInformation = state["required_information"]
    if (
        provided_information.provided_full_name
        and provided_information.provided_mobile
        and provided_information.provided_id_4_digits
    ):
        return "info all collected"

    else:
        return "not fulfill"

"""定義 Chain"""

collect_info_chain = collect_info_prompt | llm.with_structured_output(RequiredInformation)
get_information_chain = assistant_prompt | llm

# 定義助理節點函數
def assistant_chain_func(state: AssistantGraphState) -> Dict[str, Any]:
    get_information_chain = assistant_prompt | llm

    res = get_information_chain.invoke(
        {
            "user_question": state["user_question"],
            "provided_required_information": state["required_information"],
            "messages": state["messages"] if "messages" in state else [],
        }
    )

    # 更新狀態
    updated_state = state.copy()
    updated_state["messages"] = state.get("messages", []) + [res]

    return updated_state

## 檢查收集的資訊是否充足
def combine_required_info(info_list: List[RequiredInformation]) -> RequiredInformation:
    info_list = [info for info in info_list if info is not None]

    if len(info_list) == 1:
        return info_list[0]
    combined_info = {}
    for info in info_list:
        for key, value in info.dict().items():
            if value is not None:
                combined_info[key] = value
    return RequiredInformation(**combined_info)

# 收集用戶資訊，驗證，並更新 AssistantGraphState
def collect_info_chain_func(state: AssistantGraphState) -> AssistantGraphState:
    """
    收集用戶資訊，驗證，並更新 AssistantGraphState。

    參數:
    state (AssistantGraphState): 當前的助理狀態

    返回:
    AssistantGraphState: 更新後的助理狀態

    說明:
    1. 從標準輸入獲取用戶提供的資訊
    2. 調用 collect_info_chain 處理用戶輸入和當前狀態
    3. 驗證新收集的資訊
    4. 合併新收集的資訊與現有資訊（如果存在）
    5. 更新並返回新的狀態，包括更新後的必要資訊和消息歷史
    """
    # 從標準輸入獲取用戶資訊
    information_from_stdin = str(input("\n輸入用戶資訊：\n"))

    # 調用 collect_info_chain 處理用戶輸入
    response = collect_info_chain.invoke(
        {
          "user_question": state["user_question"],
          "provided_required_information": information_from_stdin,
          "messages": state["messages"],
        }
    )

    # 合併新收集的資訊與現有資訊
    if "required_information" in state:
        required_info = combine_required_info(
            info_list=[response, state.get("required_information")]
        )
    else:
        required_info = response

    # 返回更新後的狀態
    return {
        "required_information": required_info,
        "messages": [HumanMessage(content=information_from_stdin)],
    }

# 定義回應建構器節點函數
def response_builder_func(state: Dict[str, Any]) -> Dict[str, Any]:
    # 提取用戶資訊
    user_info = state.get("required_information", {})
    if hasattr(user_info, 'dict'):
        user_info = user_info.dict()

    # 提取對話歷史
    chat_history = state.get("messages", [])
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history if hasattr(msg, 'type') and hasattr(msg, 'content')])

    # 生成總結回應
    response_chain = response_builder_prompt | llm
    summary_response = response_chain.invoke({
        "user_info": user_info,
        "chat_history": chat_history_str
    })

    # 更新狀態
    updated_state = state.copy()
    updated_state["final_response"] = summary_response.content
    updated_state["messages"] = chat_history + [summary_response]

    return updated_state

# 測試資訊收集 Chain 的函數
def test_collect_info(user_input, messages = [], collected_info=None):
    if collected_info is None:
        collected_info = RequiredInformation()
    result = collect_info_chain.invoke({
        "user_question": user_input,
        "provided_required_information": collected_info,
        "messages": [],
    })
    print(f"用戶輸入: {user_input}")
    print(f"AI回應: {result}")
    print()


"""定義流程圖"""
def build_graph():
    # 定義節點名稱
    ASSISTANT_NODE = "assistant_node"
    COLLECT_INFO_NODE = "collect_info_node"
    RESPONSE_BUILDER_NODE = "response_builder_node"

    workflow = StateGraph(AssistantGraphState)

    # 添加節點
    workflow.add_node(ASSISTANT_NODE, assistant_chain_func)
    workflow.add_node(COLLECT_INFO_NODE, collect_info_chain_func)
    workflow.add_node(RESPONSE_BUILDER_NODE, response_builder_func)

    # 添加邊
    workflow.add_edge(START, "assistant_node" )
    workflow.add_edge("assistant_node", "collect_info_node")
    workflow.add_conditional_edges(
        "collect_info_node",
        provided_all_details,
        {
            "info all collected": "response_builder_node",
            "not fulfill": "assistant_node"
        }
    )
    workflow.add_edge("response_builder_node", END)

    # 編譯
    graph = workflow.compile()

    return graph

if __name__ == "__main__":
    
    # print("=== 測試資訊收集 Chain ===")
    # test_collect_info("我的名字是張小明")
    # test_collect_info("我的電話是0912345678")
    # test_collect_info("我的身分證末四碼是5678")

    graph = build_graph()

    # create_mermaid(graph)

    import rich

    init_state = AssistantGraphState(
        user_question="我想訂購高鐵票",
        required_information=RequiredInformation(),
        messages=[],
    )

    for output in graph.stream(
        init_state,
        config={"configurable": {"thread_id": 888}}
    ):
        for key, value in output.items():
            if "messages" in value:
                try:
                    last_msg = value["messages"][-1]
                    last_msg.pretty_print()
                except Exception as e:
                    print(f"last_msg:{last_msg}")