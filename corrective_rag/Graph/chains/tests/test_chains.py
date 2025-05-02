from corrective_rag.Graph.chains.retrieved_docs_grader import DocumentGrade, grading_chain
from corrective_rag.ingestion_to_chromadb import retriever
from dotenv import load_dotenv
load_dotenv()

def test_retrieved_docs_grader_yes() -> None:
    question = "Agent Memory"
    docs = retriever.invoke(question)
    doc_content = docs[0].page_content
    grade: DocumentGrade = grading_chain.invoke(input={"question": question, "document": doc_content})

    assert grade.binary_grade.lower() == "yes"

def test_retrieved_docs_grader_no() -> None:
    question = "Agent memory"
    docs = retriever.invoke(question)
    doc_content = docs[0].page_content
    grade: DocumentGrade = grading_chain.invoke(input={"question": "How to make pizza", "document": doc_content})

    assert grade.binary_grade.lower() == "no"

