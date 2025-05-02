from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

loaded_raw_documents = [WebBaseLoader(url).load() for url in urls]
loaded_raw_docs = [doc[0] for doc in loaded_raw_documents]
print(f"loaded {len(loaded_raw_docs)} number of documents from web")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(loaded_raw_docs)

print(f"Split into {len(split_docs)} documents for ingesting")

embeddings = OpenAIEmbeddings()

# vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)
# vector_store.save_local("Faiss_store")

print("---------Ingestion done---------------")
retriever = FAISS.load_local("Faiss_store", embeddings, allow_dangerous_deserialization=True).as_retriever()
