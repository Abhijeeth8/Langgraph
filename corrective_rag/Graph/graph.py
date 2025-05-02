from corrective_rag.Graph.consts import RETRIEVE, GRADE, WEBSEARCH, GENERATE
from corrective_rag.Graph.nodes import document_grader, generator_node, retrieve_docs, web_search

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from corrective_rag.Graph.state import GraphState

builder = StateGraph(GraphState)
builder.add_node(RETRIEVE, retrieve_docs.retrieve_documents)
builder.add_node(GRADE, document_grader.grade_docs)
builder.add_node(WEBSEARCH, web_search.web_search)
builder.add_node(GENERATE, generator_node.generate_node)

builder.set_entry_point(RETRIEVE)

def is_websearch_needed(graph_state:GraphState):
    print("---Checking if web search is needed------")
    web_search_needed = graph_state["web_search"]
    if web_search_needed:
        print("Yes websearch is needed")
        return WEBSEARCH
    print("No websearch is not needed")
    return GENERATE

builder.add_edge(RETRIEVE, GRADE)
builder.add_conditional_edges(GRADE, is_websearch_needed, {WEBSEARCH:WEBSEARCH, GENERATE: GENERATE})
builder.add_edge(WEBSEARCH, GENERATE)
builder.add_edge(GENERATE, END)

final_graph = builder.compile()

print(final_graph.get_graph().draw_mermaid())
print(final_graph.get_graph().draw_ascii())


