from langchain.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from corrective_rag.Graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()

from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI()

rag_chain = rag_prompt | llm | StrOutputParser()


