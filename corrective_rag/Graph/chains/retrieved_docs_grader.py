from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from corrective_rag.Graph.state import GraphState
from typing import List, Any, Dict

from corrective_rag.ingestion_to_chromadb import retriever


class DocumentGrade(BaseModel):
    """Binary grade for the relevancy of the retrieved documents"""

    binary_grade: str = Field(description="Whether the document is relevant to the question, 'yes' or 'no'")


grader_llm = ChatOpenAI(temperature=0).with_structured_output(DocumentGrade)

grading_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grading_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grading_system_prompt),
        ("human", "Retrieved documents \n\n document: \n{document} \n\n user question: {question}")
    ]
)

grading_chain = grading_prompt | grader_llm

if __name__ == "__main__":
    docs = retriever.invoke("Agent memory")
    res = grading_chain.invoke({"question": "Agent memory", "document": docs[0]})
    print(res)


