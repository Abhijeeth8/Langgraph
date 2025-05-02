from corrective_rag.Graph.chains.retrieved_docs_grader import grading_chain, DocumentGrade
from corrective_rag.Graph.state import GraphState
from typing import Dict, Any, List

from corrective_rag.ingestion_to_chromadb import retriever


def grade_docs(graph_state:GraphState) -> Dict[str, Any]:
    """Checks whether the retrieved documents are relevant to the search question.
    If not then we will set the web_search flag to true to search the web to get information

    Args: graph_state (dict) : The current state of the graph
    Returns: graph_state (dict) : The updated state of the graph with filtered out documents and web-search flag value"""

    print("-------Checking the relevancy of the retrieved docs---------")
    question = graph_state["question"]
    docs = graph_state["documents"]

    web_search = False
    filtered_docs_relevant = []

    for doc in docs:
        grade = grading_chain.invoke({"question": question, "document": doc})

        if grade.binary_grade == "yes":
            print("--------DOC RELEVANT-----------")
            filtered_docs_relevant.append(doc)
        else:
            print("--------DOC NOT RELEVANT-----------")
            web_search=True
            continue

    return {"question": question, "documents": filtered_docs_relevant, "web_search": web_search}

if __name__ == "__main__":
    question = "Agent memory"
    docs = retriever.invoke(question)
    doc_content = docs[0].page_content
    res = grade_docs({"question": question, "documents": doc_content})
    # grade: DocumentGrade = grading_chain.invoke(input={"question": question, "document": doc_content})