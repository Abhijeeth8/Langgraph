from dotenv import load_dotenv
load_dotenv()

from typing import List, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from Chains import tweet_gen_chain, critique_chain
from langgraph.graph import END, MessageGraph

REVISE = "revise"
CRITIQUE = "critique"


def tweet_gen_node(state:List[BaseMessage]) -> List[BaseMessage]:
    return tweet_gen_chain.invoke({"messages": state})

def critique_node(state:List[BaseMessage]) -> List[BaseMessage]:
    critique = critique_chain.invoke({"messages": state})
    return [HumanMessage(content=critique.content)]

graph = MessageGraph()

graph.add_node(REVISE, tweet_gen_node)
graph.add_node(CRITIQUE, critique_node)

graph.set_entry_point(REVISE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return CRITIQUE

graph.add_conditional_edges(REVISE, should_continue)
graph.add_edge(CRITIQUE, REVISE)

real_graph = graph.compile()
print(real_graph.get_graph().draw_mermaid())
# real_graph.get_graph().draw_png(output_file_path="graph.png")



if __name__ == "__main__":
    print("Hello Langgraph's reflection agent")

    tweet = HumanMessage(
            content = """Make this tweet better:
We’re moving from just prompting to reasoning, planning, and acting.
LangGraph, AutoGen, and CrewAI are changing the game.
Can’t wait to build with truly autonomous systems.
#AI #LLM #AgenticAI #LangChain #AutonomousAgents
            """)

    result = real_graph.invoke(tweet)
    print(result[-1].content)

