import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI  # still used for answering
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ✅ Local embeddings (FREE, no OpenAI quota needed)
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ LangChain v1.x fix: chains moved to langchain-classic
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


def load_pdfs_from_folder(folder: str):
    docs = []
    if not os.path.exists(folder):
        raise SystemExit(f"Folder not found: {folder}")

    for fn in os.listdir(folder):
        if fn.lower().endswith(".pdf"):
            path = os.path.join(folder, fn)
            loader = PyPDFLoader(path)
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = fn
            docs.extend(pdf_docs)
    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _local_embeddings():
    # Small, fast, good default for RAG
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_faiss_index(chunks, index_dir: str = "faiss_index"):
    embeddings = _local_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)
    return db


def load_faiss_index(index_dir: str = "faiss_index"):
    embeddings = _local_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


LEGAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Legal AI Assistant. Use the provided context only.\n"
            "Rules:\n"
            "1) DO NOT invent laws, citations, or clauses.\n"
            "2) If context is insufficient, say you don't know and ask for missing info.\n"
            "3) Always include a short 'Not legal advice' disclaimer.\n"
            "4) Output format: Summary, Relevant Excerpts (with sources), Risks, Next Steps.\n"
            "5) Be jurisdiction-aware: if jurisdiction is unknown, ask.\n",
        ),
        (
            "human",
            "User question: {input}\n\n"
            "Context:\n{context}\n",
        ),
    ]
)


def format_sources(source_docs):
    seen = set()
    lines = []
    for d in source_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        if page is not None:
            lines.append(f"- {src} (page {page + 1})")
        else:
            lines.append(f"- {src}")
    return "\n".join(lines) if lines else "- None"


def make_legal_rag_chain(db):
    # ⚠️ Needs OpenAI quota for answering
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    doc_chain = create_stuff_documents_chain(llm, LEGAL_PROMPT)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain


def main():
    data_dir = "data"
    index_dir = "faiss_index"

    if os.path.exists(index_dir):
        db = load_faiss_index(index_dir)
    else:
        docs = load_pdfs_from_folder(data_dir)
        if not docs:
            raise SystemExit("No PDFs found in ./data. Add documents and rerun.")
        chunks = chunk_docs(docs)
        db = build_faiss_index(chunks, index_dir=index_dir)

    rag = make_legal_rag_chain(db)

    print("✅ Legal RAG Assistant ready. Type your question. (type 'exit' to quit)")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = rag.invoke({"input": q})

        print("\n--- Answer ---\n")
        print(result.get("answer", ""))

        print("\n--- Sources used ---\n")
        print(format_sources(result.get("context", [])))


if __name__ == "__main__":
    main()
