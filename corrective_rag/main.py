from dotenv import load_dotenv

load_dotenv()

from Graph.graph import final_graph

if __name__ == "__main__":
    print("Hello langgraph's advanced RAG")
    result = final_graph.invoke(input={"question":"What is Agent Memory?"})
    print(result["generation"])