from langchain_openai import ChatOpenAI # helps to connect to the LLM - important role as a wrapper around the model
from langchain_core.prompts import ChatPromptTemplate # helps to create prompts for the LLM
from langchain_core.output_parsers import StrOutputParser # helps to parse the output from the LLM and get cleaner output
from langchain_community.embeddings import HuggingFaceEmbeddings # helps to create embeddings for the text data\
from langchain_community.document_loaders import PyPDFLoader # helps to load the pdf documents and convert them into text data
from langchain_text_splitters import RecursiveCharacterTextSplitter # helps to split the text data into smaller chunks for better processing
from langchain_community.vectorstores import FAISS # helps to create a vector store for the embeddings and perform similarity search
from langchain_core.runnables import RunnablePassthrough # helps to create a runnable that can be used to pass the input data through the chain without any modification
from langchain.agents import initialize_agent, AgentType, tool # helps to create an agent that can use the tools to perform specific tasks based on the user's query
api_key = "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"

llm = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = api_key,
    openai_api_base = "https://openrouter.ai/api/v1",
)

output_parser = StrOutputParser()

loader = PyPDFLoader(r"D:\Agentic_Course\Agents\ai_agents\2603.09858v1.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
do = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(do, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

@tool
def paper_search(query : str) -> str:
    """This tool is used to search for relevant information from the paper based on the user's query."""
    docs = retriever.invoke(query)
    context =  "\n".join([doc.page_content for doc in docs])
    return context

@tool
def calculator(query : str) -> int:
    """This tool is used to perform mathematical calculations based on the user's query."""
    result = eval(query)
    return result

tools = [paper_search, calculator]
agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

while True:
    question = input("Ask a question about the paper or perform a calculation: ")
    if question.lower() == "exit":
        break
    response = agent.run(question)
    print(response)