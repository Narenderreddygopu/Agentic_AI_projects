from langchain_openai import ChatOpenAI # helps to connect to the LLM - important role as a wrapper around the model
from langchain_core.prompts import ChatPromptTemplate # helps to create prompts for the LLM
from langchain_core.output_parsers import StrOutputParser # helps to parse the output from the LLM and get cleaner output
from langchain_community.embeddings import HuggingFaceEmbeddings # helps to create embeddings for the text data\
from langchain_community.document_loaders import PyPDFLoader # helps to load the pdf documents and convert them into text data
from langchain_text_splitters import RecursiveCharacterTextSplitter # helps to split the text data into smaller chunks for better processing
from langchain_community.vectorstores import FAISS # helps to create a vector store for the embeddings and perform similarity search
from langchain_core.runnables import RunnablePassthrough # helps to create a runnable that can be used to pass the input data through the chain without any modification
from langchain_community.document_loaders import ArxivLoader # helps to load the arxiv documents and convert them into text data

api_key = "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"

llm = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = api_key,
    openai_api_base = "https://openrouter.ai/api/v1",
)

output_parser = StrOutputParser()

doc = PyPDFLoader(r"D:\Agentic_Course\Agents\ai_agents\2603.09858v1.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(doc)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
template = """
You are a Research Assistant. You will be given a question and some context. Use the context to answer the question. If you don't know the answer, say you don't know.
Context: {context}  
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(template)
doc_chain = (   
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | output_parser
)   
user_question = input("Ask a question about the paper: ")
result = doc_chain.invoke(user_question)
print(result)