from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph,END
from typing import TypedDict


api_key = "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"

llm = ChatOpenAI(
    model = "nvidia/nemotron-3-super-120b-a12b:free",
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

def router(state: GraphState):
    print("\n---Router Node")
    question = state["question"]
    router_prompt = f"""
Deceide whether the question requires retriving information from the research paper.

If it requires the paper respond with : 
retrieve

If it can be answered directly respond with : 
direct

Question : {question}
"""
    decision = llm.invoke(router_prompt).content.lower()

    if "retrieve" in decision :
        print("going to retrieve node")
        return "go_to_retrieve_node"
    else :
        print("going to direct node")
        return "go_to_direct_node"
    
def retrieve(state: GraphState):
    print("\n---Retrieve Node")
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n".join([i.page_content for i in docs])
    return {"context": context}

def generate(state:GraphState):
    print("\n---Generate Node")
    question = state["question"]
    context = state["context"]

    prompt = rag_prompt.format(
        context = context,
        question = question
    )
    response = llm.invoke(prompt)
    return {"answer" : response.content}

def direct_answer(state:GraphState):
    print("\n---Direct Answer Node")
    question = state["question"]
    response = llm.invoke(question)
    return {"answer" : response.content}


CG = StateGraph(GraphState)

CG.add_node("retrieve",retrieve)
CG.add_node("generate",generate)
CG.add_node("direct_answer",direct_answer)

CG.set_conditional_entry_point(
    router,
    {
        "go_to_retrieve_node" : "retrieve",
        "go_to_direct_node" : "direct_answer"
    }
)

CG.add_edge("retrieve","generate")
CG.add_edge("generate",END)
CG.add_edge("direct_answer",END)


Graph = CG.compile()

while True : 
    user_question = input("\nWhat is your question : ")

    if user_question.lower() == "exit":
        break

    result = Graph.invoke({"question" : user_question})
    print("\nfinal answer : ")
    print(result["answer"])