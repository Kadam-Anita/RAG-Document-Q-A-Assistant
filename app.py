"""
RAG Document Q&A Assistant
Based on the uploaded notebook: PDF retrieval + relevance evaluation +
sentence filtering + answer generation (no external web-search API).
Run: streamlit run app.py
"""

import os
import re
import tempfile
from typing import List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
st.title("📚 RAG Document Q&A Assistant")
st.caption("Ask questions about your PDFs. Relevant documents are retrieved and checked before an answer is generated.")

UPPER_TH = 0.7
LOWER_TH = 0.3


class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    answer: str


class DocEvalScore(BaseModel):
    score: float
    reason: str


class KeepOrDrop(BaseModel):
    keep: bool


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_uploaded_pdfs(uploaded_files):
    documents = []
    for uploaded in uploaded_files:
        # PyPDFLoader expects a path, so write each upload to a temporary PDF.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            pdf_docs = PyPDFLoader(path).load()
            for doc in pdf_docs:
                doc.metadata["source"] = uploaded.name
            documents.extend(pdf_docs)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return documents


def build_graph(retriever, llm):
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval evaluator for RAG. You will be given ONE retrieved "
         "chunk and a question. Return a relevance score in [0.0, 1.0]. "
         "1.0 means the chunk is sufficient to answer fully/mostly; 0.0 means irrelevant. "
         "Be conservative with high scores. Return a short reason."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ])
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

    filter_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict relevance filter. Return keep=true only if the sentence directly "
         "helps answer the question. Judge only the provided sentence."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ])
    filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful document Q&A assistant. Answer using ONLY the provided context. "
         "If the context is empty or insufficient, clearly say that the uploaded documents "
         "do not contain enough information. Do not invent facts."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ])

    def retrieve_node(state: State) -> State:
        return {"docs": retriever.invoke(state["question"])}

    def eval_each_doc_node(state: State) -> State:
        scores, good = [], []
        reasons = []
        for d in state["docs"]:
            result = doc_eval_chain.invoke({
                "question": state["question"],
                "chunk": d.page_content,
            })
            scores.append(float(result.score))
            reasons.append(result.reason)
            if result.score > LOWER_TH:
                good.append(d)

        if any(score > UPPER_TH for score in scores):
            verdict = "CORRECT"
            reason = f"At least one retrieved chunk scored above {UPPER_TH}."
        elif scores and all(score < LOWER_TH for score in scores):
            verdict, good = "INCORRECT", []
            reason = f"All retrieved chunks scored below {LOWER_TH}."
        else:
            verdict = "AMBIGUOUS"
            reason = "Retrieved chunks have mixed or borderline relevance."
        return {"good_docs": good, "verdict": verdict, "reason": reason}

    def refine_node(state: State) -> State:
        # Without a web-search provider, use only relevant chunks from uploaded PDFs.
        # If retrieval is judged incorrect, good_docs will be empty and the LLM
        # will be instructed to say the documents do not contain enough information.
        docs_to_use = state["good_docs"]

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", context).strip())
        sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]

        kept = []
        # Avoid spending excessive local-model time evaluating huge context.
        for sentence in sentences[:80]:
            try:
                if filter_chain.invoke({
                    "question": state["question"], "sentence": sentence
                }).keep:
                    kept.append(sentence)
            except Exception:
                # If structured filtering fails, do not silently lose all context.
                continue

        refined = "\n".join(kept).strip()
        if not refined:
            refined = context
        return {
            "strips": sentences,
            "kept_strips": kept,
            "refined_context": refined,
        }

    def generate_node(state: State) -> State:
        result = (answer_prompt | llm).invoke({
            "question": state["question"],
            "context": state["refined_context"],
        })
        return {"answer": result.content}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("eval_each_doc", eval_each_doc_node)
    graph.add_node("refine", refine_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "eval_each_doc")
    # No Tavily/API key: all questions continue through the document-only RAG path.
    graph.add_edge("eval_each_doc", "refine")
    graph.add_edge("refine", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "gemma3:1b"))
    st.caption("Ollama must be installed and running locally. This version answers using uploaded PDFs only; no web-search API key is needed.")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )
    build_button = st.button("Build / Refresh document index", type="primary", use_container_width=True)

if build_button:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        try:
            with st.spinner("Reading PDFs, splitting text, and creating the FAISS index..."):
                docs = load_uploaded_pdfs(uploaded_files)
                if not docs:
                    raise ValueError("No readable pages were found in the uploaded PDFs.")
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=900, chunk_overlap=150
                ).split_documents(docs)
                for doc in chunks:
                    doc.page_content = doc.page_content.encode(
                        "utf-8", "ignore"
                    ).decode("utf-8", "ignore")
                vector_store = FAISS.from_documents(chunks, get_embeddings())
                st.session_state["retriever"] = vector_store.as_retriever(
                    search_type="similarity", search_kwargs={"k": 2}
                )
                st.session_state["document_count"] = len(docs)
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["source_names"] = [f.name for f in uploaded_files]
            st.success("Document index created successfully.")
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")

if "retriever" not in st.session_state:
    st.info("👈 Upload one or more PDFs in the sidebar, then click **Build / Refresh document index**.")
    st.markdown("""
    **What this app does**
    1. Reads PDF documents and splits them into overlapping text chunks.
    2. Converts chunks into embeddings and indexes them in FAISS.
    3. Retrieves the four most similar chunks for a question.
    4. Uses the local LLM to judge relevance and filters context.
    5. If retrieval is weak or ambiguous, uses only relevant retrieved PDF chunks.
    6. If the PDFs do not contain enough information, says so rather than searching the web.
    """)
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("PDF pages loaded", st.session_state.get("document_count", 0))
col2.metric("Text chunks indexed", st.session_state.get("chunk_count", 0))
col3.metric("Retrieved per question", 2)

question = st.text_area(
    "Ask a question about your documents",
    placeholder="Example: What are the steps to deploy an ML model?",
    height=90,
)
ask_button = st.button("Get answer", type="primary")

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner(f"Retrieving evidence and generating an answer with {model_name}..."):
                llm = ChatOllama(model=model_name, temperature=0)
                app = build_graph(st.session_state["retriever"], llm)
                result = app.invoke({
                    "question": question.strip(),
                    "docs": [],
                    "good_docs": [],
                    "verdict": "",
                    "reason": "",
                    "strips": [],
                    "kept_strips": [],
                    "refined_context": "",
                    "answer": "",
                })
            st.subheader("Answer")
            st.write(result["answer"])
            with st.expander("How the system reached this answer"):
                st.write("**Retrieval verdict:**", result["verdict"])
                st.write("**Reason:**", result["reason"])
                st.write(f"**Retrieved chunks:** {len(result['docs'])}")
                st.write(f"**Relevant document chunks kept:** {len(result['good_docs'])}")
                st.write(f"**Sentences kept after filtering:** {len(result['kept_strips'])}")
                if result["docs"]:
                    st.markdown("**Retrieved document excerpts**")
                    for i, doc in enumerate(result["docs"], start=1):
                        st.markdown(f"**Chunk {i} — {doc.metadata.get('source', 'PDF')}**")
                        st.write(doc.page_content[:1200])
        except Exception as exc:
            st.error(
                f"Something went wrong: {exc}\n\n"
                "Check that Ollama is running, the selected model is installed, and dependencies are installed."
            )
