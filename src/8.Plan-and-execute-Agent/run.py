import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import asyncio
from typing import Annotated, List, Tuple, TypedDict, Union, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from llm import LLMManager
import operator
from utils.graph2mermaid import create_mermaid # for saving mermaid code
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """A simple search tool that returns mock results for demonstration purposes."""
    query_lower = query.lower()

    # 檢查是否是關於2024年奧運會羽毛球男子雙打的查詢
    if any(keyword in query_lower for keyword in ["2024", "olympic", "羽毛球", "badminton", "men's doubles", "男子雙打", "champion", "gold medal"]):
        return """Search results for '2024 Olympics badminton men's doubles gold medal':

1. **Official Olympic Results** - International Olympic Committee (IOC)
   - Gold Medal: Li Junhui and Liu Yuchen (China)
   - Silver Medal: Liang Weikeng and Wang Chang (China)
   - Bronze Medal: Aaron Chia and Soh Wooi Yik (Malaysia)

2. **Badminton World Federation (BWF) Records**
   - 2024 Paris Olympics Men's Doubles Final
   - Winners: Li Junhui / Liu Yuchen (CHN)
   - Score: 2-0 vs Liang Weikeng / Wang Chang (CHN)

3. **BBC Sports Coverage**
   - China dominates badminton at Paris 2024
   - Li Junhui and Liu Yuchen claim men's doubles gold
   - First Olympic gold for the Chinese pair

4. **ESPN Olympics Report**
   - Chinese sweep in badminton continues
   - Liu Yuchen and Li Junhui win dramatic final"""

    # 默認模擬結果
    return f"Mock search results for '{query}': 1. Result 1, 2. Result 2, 3. Result 3."

tools = [search_tool]

# 步驟 1：定義狀態
class PlanExecute(TypedDict):
    input: str  # 原始輸入
    plan: List[str]  # 當前計劃
    past_steps: Annotated[List[Tuple], operator.add]  # 已執行的步驟
    response: str  # 最終響應


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# 步驟 2：定義語言模型
llm_manager = LLMManager()
# prompt = hub.pull("wfh/react-agent-executor")
# Replace with local prompt due to connection issues
prompt = """You are a helpful assistant. Use the provided tools to answer questions.

Available tools: {tools}

To use a tool, respond with the tool name and arguments in the format:
TOOL: tool_name(args)

For example:
TOOL: search(query="example")

After getting tool results, provide the final answer."""

llm = llm_manager.get_llm("chat")
agent_executor = create_react_agent(llm, tools, prompt=prompt)

"""定義提示模板"""
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Always provide at least one step in the plan.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

"""chain the prompts and llm to create the final nodes"""
planner = planner_prompt | llm.with_structured_output(Plan)
replanner = replanner_prompt | llm.with_structured_output(Act)


"""定義節點函數"""
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    # make past_steps as list and append task, agent_response["messages"][-1].content)
    past_steps_list = []
    past_steps_list.append(task)
    past_steps_list.append(agent_response["messages"][-1].content)
    return {
        "past_steps": past_steps_list,
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"

# 步驟 4：構建圖
def build_graph():
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    graph = workflow.compile()

    return graph


# 步驟 6：實現聊天界面
async def chat_interface(graph):
    import rich
    
    config = {"recursion_limit": 10}
    inputs = {"input": "2024 奧運男子組羽毛球雙打冠軍是誰?"}
    async for event in graph.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
            if k == "response":
                rich.print("Agent says: ", v)

if __name__ == "__main__":
    # 運行聊天界面
    
    
    graph = build_graph()
    asyncio.run(chat_interface(graph))
    # create_mermaid(graph)