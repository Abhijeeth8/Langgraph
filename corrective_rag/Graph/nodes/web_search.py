import os

from corrective_rag.Graph.state import GraphState
from typing import Dict, List, Any
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(graph_state:GraphState) -> Dict[str, Any]:

    print("-------Searching Web---------")
    question = graph_state["question"]
    relevant_docs = graph_state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})["results"]

    web_search_docs = "\n---------\n".join([tavily_search_result["content"] for tavily_search_result in tavily_results])

    if relevant_docs is not None:
        relevant_docs.append(web_search_docs)
    else:
        relevant_docs = [web_search_docs]

    return {"question": question, "documents": relevant_docs}

if __name__ == "__main__":
    res = web_search(graph_state={"question": "Agent memory", "documents": None})
    print(res)