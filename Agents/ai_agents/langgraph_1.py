from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph,END
from typing import TypedDict

api_key =  "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"

llm = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = api_key,
    openai_api_base = "https://openrouter.ai/api/v1",
)

paper_id = "1706.03762"

loader = ArxivLoader(query = paper_id,load_max_docs = 1)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

template = """
You are a Research Assistant. You will be given a question and some context. Use the context to answer the question. If you don't know the answer, say you don't know.
Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(template)

class GraphState(TypedDict):
    question: str
    context: list
    answer: str

def retrieve(state: GraphState):
    print("by retrieve node")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"context" : docs}

def generate(state: GraphState):
    print("by generate node")
    question = state["question"]
    context = state["context"]
    formatted_prompt = rag_prompt.format(context = context,question = question)
    response = llm.invoke(formatted_prompt)
    return {"answer": response.content}

graph = StateGraph(GraphState)
graph.add_node("retrieve",retrieve)
graph.add_node("generate",generate)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve","generate")

graph.add_edge("generate",END)

GG = graph.compile()

user_question = input("ask a question : ")
result = GG.invoke({"question" : user_question})
print(result["answer"])