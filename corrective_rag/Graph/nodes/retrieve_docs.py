from dotenv import load_dotenv

from typing import Dict, Any
from corrective_rag.Graph.state import GraphState
from corrective_rag.ingestion_to_chromadb import retriever
from langchain_community.vectorstores import FAISS

load_dotenv()

def retrieve_documents(graph_state:GraphState) -> Dict[str, Any]:
    print("_________Retrieving documents related to the question from the vector store and updating the state___________")
    retrieved_docs = retriever.invoke(graph_state["question"])
    # graph_state["documents"].append(retrieved_docs.page_content)
    return {"question": graph_state["question"], "documents": retrieved_docs}

if __name__ == "__main__":
    print("_________Retrieving documents related to the questiom from the vector store and updating the state___________")