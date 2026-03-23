from langchain_openai import ChatOpenAI # helps to connect to the LLM - important role as a wrapper around the model
from langchain_core.prompts import ChatPromptTemplate # helps to create prompts for the LLM
from langchain_core.output_parsers import StrOutputParser # helps to parse the output from the LLM and get cleaner output
from langchain_community.embeddings import HuggingFaceEmbeddings # helps to create embeddings for the text data\
from langchain_community.document_loaders import PyPDFLoader # helps to load the pdf documents and convert them into text data
from langchain_text_splitters import RecursiveCharacterTextSplitter # helps to split the text data into smaller chunks for better processing
from langchain_community.vectorstores import FAISS # helps to create a vector store for the embeddings and perform similarity search
from langchain_core.runnables import RunnablePassthrough # helps to create a runnable that can be used to pass the input data through the chain without any modification

api_key = "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"

llm = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = api_key,
    openai_api_base = "https://openrouter.ai/api/v1",
)

output_parser = StrOutputParser()


# context = 1 to 300
# chunk 1 = 1 to 100
# chunk 2 = 80 to 180
local_context = "The planet pluto was recently considered as a planet but now it is considered as a dwarf planet because it does not meet all the criteria to be classified as a planet. It was discovered in 1930 and was named after the Roman god of the underworld. Pluto has a very eccentric orbit and takes about 248 years to complete one orbit around the sun. It has five known moons, the largest of which is Suraj."
text_splitter_1 = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter_1.split_text(local_context)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(chunks, embeddings)
retriever = vector_db.as_retriever()
template = """
ANswer the question based on the following context:
{cont}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"cont": retriever , "question" : RunnablePassthrough() } | prompt | llm | output_parser
)

user_question = input("Ask a question about pluto: ")
result = chain.invoke(user_question)
print(result)