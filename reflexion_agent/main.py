from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph import graph
from langgraph.graph import END
from langchain_core.messages import BaseMessage, ToolMessage
from reflexion_agent.Chains import generate_ans_chain, revise_chain
from reflexion_agent.tools_executor import tool_node

load_dotenv()

DRAFT = "draft"
EXEC_TOOLS = "exec_tools"
REVISE = "revise"


builder = graph.MessageGraph()

builder.add_node(DRAFT, generate_ans_chain)
builder.add_node(EXEC_TOOLS, tool_node)
builder.add_node(REVISE, revise_chain)
builder.set_entry_point(DRAFT)
builder.add_edge(DRAFT, EXEC_TOOLS)
builder.add_edge(EXEC_TOOLS, REVISE)

MAX_ITERATIONS = 2
def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return EXEC_TOOLS

builder.add_conditional_edges(REVISE, event_loop)


graph = builder.compile()
print(graph.get_graph().draw_ascii())


if __name__ == "__main__":
    print("Hello langgraph's reflexion agent")

    human_question = "Write about Retrieval Augmentation Generation in LLMs. And also list some related sources related to that."

    result = graph.invoke(input=human_question)

    print(result)

#
# from typing import List
#
# from dotenv import load_dotenv
#
# load_dotenv()
# from langchain_core.messages import BaseMessage, ToolMessage
# from langgraph.graph import END, MessageGraph
#
# from Chains import first_responder, revisor
# from tools_executor import tool_node
#
# MAX_ITERATIONS = 2
# builder = MessageGraph()
# builder.add_node("draft", first_responder)
# builder.add_node("execute_tools", tool_node)
# builder.add_node("revise", revisor)
# builder.add_edge("draft", "execute_tools")
# builder.add_edge("execute_tools", "revise")
#
#
# def event_loop(state: List[BaseMessage]) -> str:
#     count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
#     num_iterations = count_tool_visits
#     if num_iterations > MAX_ITERATIONS:
#         return END
#     return "execute_tools"
#
#
# builder.add_conditional_edges("revise", event_loop)
# builder.set_entry_point("draft")
# graph = builder.compile()
#
# # print(graph.get_graph().draw_mermaid())
# # print(graph.get_graph().draw_ascii())
#
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
#
# res = graph.invoke(
#     "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
# )
# print(res[-1].tool_calls[0]["args"]["answer"])
# print(res)
