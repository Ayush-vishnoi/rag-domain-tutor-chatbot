import os
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

TUTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert NLP course tutor. Use the course materials below as your primary source to answer the student's question.
Be pedagogical: explain concepts clearly, give examples, and reference specific modules or chapters when relevant.
If the topic is genuinely not covered at all in the materials, say so briefly — but always try to help using what is available.

Course Materials:
---------------
{context}
---------------

Student Question: {question}

Answer:"""
)

def load_vectorstore(corpus_dir: str = "./syllabus_docs"):
    documents = []
    for fname in os.listdir(corpus_dir):
        if fname.endswith(".md"):
            loader = TextLoader(os.path.join(corpus_dir, fname), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = fname
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, embeddings

def build_chain(vectorstore, groq_api_key: str):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | TUTOR_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def check_hallucination(answer: str, source_docs: list, embeddings, threshold: float = 0.75):
    context = " ".join([d.page_content for d in source_docs])
    answer_emb = embeddings.embed_query(answer)
    context_emb = embeddings.embed_query(context)
    score = cosine_similarity(answer_emb, context_emb)
    return {
        "faithfulness_score": round(score, 4),
        "is_hallucination": score < threshold
    }
