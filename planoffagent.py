import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

# -------------------------------------------------------------------
# Streamlit Setup
# -------------------------------------------------------------------
st.set_page_config(page_title="Offline Multi-Agent PDF RAG", page_icon="🤖")
st.title("🤖 Offline Multi-Agent PDF RAG + Planner")

# -------------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load summarizer model once
if "summarizer" not in st.session_state:
    device = 0 if torch.cuda.is_available() else -1
    st.session_state.summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device
    )


# -------------------------------------------------------------------
# PDF Processing Helpers
# -------------------------------------------------------------------
def extract_text(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=1000, chunk_overlap=150):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def build_vectorstore(chunks):
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=chunks, embedding=model)


# -------------------------------------------------------------------
# Multi-Agent System
# -------------------------------------------------------------------

# ---------------------- Agent 1: RAG Agent -------------------------
class RAGAgent:
    def search(self, query: str) -> str:
        vs = st.session_state.get("vectorstore")

        if not vs:
            return "No documents indexed. Please upload PDFs first."

        docs = vs.similarity_search(query, k=4)

        if not docs:
            return "No relevant passages found."

        return "\n\n---\n\n".join(d.page_content for d in docs)

    def answer_question(self, question: str) -> str:
        context = self.search(question)

        if "No relevant" in context:
            return context

        # Summarize retrieved context to generate the final answer
        chunks = chunk_text(context, chunk_size=800, chunk_overlap=50)
        final_answer = []

        for ch in chunks:
            summary = st.session_state.summarizer(
                ch,
                max_length=150,
                min_length=40,
                do_sample=False
            )[0]['summary_text']
            final_answer.append(summary)

        return " ".join(final_answer)


# ---------------------- Agent 2: Summarization Agent -------------------------
class SummarizationAgent:
    def summarize_full_document(self) -> str:
        full_text = st.session_state.get("full_text", "")

        if not full_text.strip():
            return "No document loaded for summarization."

        chunks = chunk_text(full_text, chunk_size=800, chunk_overlap=100)
        summaries = []

        for c in chunks:
            summary = st.session_state.summarizer(
                c, max_length=200, min_length=60, do_sample=False
            )[0]['summary_text']
            summaries.append(summary)

        return " ".join(summaries)


# ---------------------- Planner Agent -------------------------
class PlannerAgent:

    SUMMARY_KEYWORDS = ["summary", "summarize", "overview", "abstract"]

    def decide(self, user_input: str) -> str:
        """
        Decide whether to call:
        - SummarizationAgent
        - RAGAgent
        """

        if any(word in user_input.lower() for word in self.SUMMARY_KEYWORDS):
            return "summarize"

        # default → question answering
        return "rag"

    def run(self, user_input: str) -> str:
        decision = self.decide(user_input)

        if decision == "summarize":
            return SummarizationAgent().summarize_full_document()

        else:
            return RAGAgent().answer_question(user_input)


# -------------------------------------------------------------------
# Sidebar - PDF Upload
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Upload PDFs")
    files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):
        if not files:
            st.warning("Please upload files first.")
        else:
            with st.spinner("Extracting text and building index..."):
                raw = extract_text(files)
                st.session_state.full_text = raw
                chunks = chunk_text(raw)
                st.session_state.vectorstore = build_vectorstore(chunks)
            st.success("PDFs processed and indexed ✓")

# -------------------------------------------------------------------
# Main Chat Interface
# -------------------------------------------------------------------
st.write("---")
user_input = st.text_input("Ask a question or request a summary:")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    planner = PlannerAgent()

    with st.spinner("Agent reasoning..."):
        reply = planner.run(user_input)

    st.session_state.chat_history.append(("bot", reply))

# Render conversation
for role, message in st.session_state.chat_history:
    color = "#e6f7ff" if role == "user" else "#fff3e6"
    label = "You" if role == "user" else "Assistant"
    st.markdown(
        f"<div style='background:{color};padding:10px;border-radius:8px;'>"
        f"<b>{label}:</b> {message}</div>",
        unsafe_allow_html=True
    )

st.caption("Multi-Agent Offline System: Planner → (RAG Agent or Summarizer Agent)")
