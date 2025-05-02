from corrective_rag.Graph.chains.generate_using_llm import rag_chain
from corrective_rag.Graph.state import GraphState
from typing import Any, Dict


def generate_node(graph_state:GraphState) -> Dict[str, Any]:
    print("-----Generating response from the LLM-------")
    question = graph_state["question"]
    documents = graph_state["documents"]

    final_generated_result = rag_chain.invoke({"question": question, "context": documents})

    return {"question": question, "documents": documents, "generation": final_generated_result}