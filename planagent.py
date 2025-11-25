"""
Multi-agent Streamlit app: PDF RAG + Summarization using AgentSDK

Requirements:
- streamlit
- PyPDF2
- langchain_text_splitters
- langchain_community (FAISS & embeddings)
- sentence-transformers
- openai (AgentSDK v0.6.x compatible)
- faiss-cpu (or faiss)

Note: This example assumes an OpenAI-compatible AgentSDK as in your earlier sample
(`from agents import Agent, function_tool, Runner`) and `openai.OpenAI` client.
Set OPENAI_API_KEY in your environment before running.

Description:
- Two agents are registered as tools:
  1) RAG agent ("rag_tool") - retrieves top-k passages and calls the LLM to answer.
  2) Summarization agent ("summarize_tool") - performs a map-reduce summarization.
- A Planner agent ("planner_agent") decides which tool to call automatically.
- The Streamlit UI allows uploading multiple PDFs, viewing processed files, and chatting.

This file is intentionally self-contained and instrumented with helpful logging and
fallbacks.
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any

# AgentSDK imports (adjust if your AgentSDK's module paths differ)
from openai import OpenAI
from agents import Agent, function_tool, Runner

# ------------ Config --------------
st.set_page_config(page_title="Multi-Agent PDF RAG + Summarize", page_icon="📚", layout="wide")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not set in environment. Set it before running for LLM-backed agents.")

# instantiate OpenAI client (used by agents to plan + generate)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------- UI helpers ---------------
css = """<style> .user {background:#e6f7ff;padding:8px;border-radius:8px;} .bot {background:#fff3e6;padding:8px;border-radius:8px;} </style>"""
st.write(css, unsafe_allow_html=True)

user_template = '<div class="user"><b>You:</b><div>{{MSG}}</div></div>'
bot_template = '<div class="bot"><b>Assistant:</b><div>{{MSG}}</div></div>'

# -------------- Session State ---------------
if "files" not in st.session_state:
    st.session_state["files"] = {}  # filename -> raw_text
if "vectorstores" not in st.session_state:
    st.session_state["vectorstores"] = {}  # filename -> FAISS index (per-file index)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# -------------- PDF ingestion & chunking ---------------

def extract_text_from_pdf_file(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text += ptext + "\n"
        return text
    except Exception as e:
        st.warning(f"Failed to parse {getattr(uploaded_file, 'name', 'file')}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_faiss_for_chunks(chunks: List[str]):
    # uses a sentence-transformers model (small & fast) locally
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(texts=chunks, embedding=emb)
    return vs

# -------------- Agent Tools ---------------
# We register tools that will be callable by the agents/planner.

@function_tool
def rag_tool(query: str, filename: str = None, top_k: int = 4) -> str:
    """
    RAG tool: retrieve top-k passages (from a specified file or across all files) and
    call the LLM to produce an answer using the retrieved context.

    Returns a text block suitable for human consumption.
    """
    # Basic guard
    if not st.session_state["vectorstores"]:
        return "No document indexes available. Upload PDFs and process them in the sidebar."

    # choose vectorstore(s)
    if filename:
        vs = st.session_state["vectorstores"].get(filename)
        if not vs:
            return f"No vectorstore found for file '{filename}'."
        candidate_docs = vs.similarity_search(query, k=top_k)
    else:
        # search across all vectorstores and aggregate top results
        candidate_docs = []
        for fn, vs in st.session_state["vectorstores"].items():
            try:
                res = vs.similarity_search(query, k=top_k)
                candidate_docs.extend(res)
            except Exception:
                continue
        # naive dedupe: keep unique page_content
        seen = set()
        unique_docs = []
        for d in candidate_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique_docs.append(d)
        candidate_docs = unique_docs[: top_k]

    if not candidate_docs:
        return "No relevant passages found in uploaded documents."

    # Build context string
    context = "\n\n---\n\n".join([f"[SOURCE: {getattr(d, 'metadata', {}).get('source','unknown')}]" + d.page_content for d in candidate_docs])

    # Ask the model to answer using the context
    prompt = (
        "You are a helpful assistant that must answer the user's question using the provided context.\n"
        "Only use information present in the context — if the answer isn't in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\nAnswer concisely and cite the relevant source lines if possible."
    )

    # call the LLM via OpenAI client
    try:
        if client is None:
            # fallback: return context & suggestion
            return "[No OpenAI client available] Retrieved context:\n\n" + context

        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # choose an available model in your environment
            messages=[
                {"role": "system", "content": "You are a concise assistant that answers based only on provided context."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        return f"**[RAG Answer — retrieved passages used]**\n\n{text}" if text else "Agent returned no answer."
    except Exception as e:
        return f"RAG failed to call LLM: {e}\n\nRetrieved context was:\n{context}"


@function_tool
def summarize_tool(filename: str = None, chunk_size: int = 800, chunk_overlap: int = 100) -> str:
    """
    Summarization tool: performs a map-reduce style summary.
    If filename is None, summarize the concatenation of all uploaded files.
    """
    # collect text
    if filename:
        txt = st.session_state["files"].get(filename, "")
        if not txt:
            return f"No text found for file '{filename}'."
    else:
        txts = [t for t in st.session_state["files"].values() if t.strip()]
        if not txts:
            return "No uploaded documents to summarize."
        txt = "\n\n".join(txts)

    if not txt.strip():
        return "No extractable text to summarize."

    # chunk text
    chunks = chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return "No chunks produced for summarization."

    # Map step: summarize each chunk with the LLM
    chunk_summaries = []
    for i, c in enumerate(chunks):
        prompt = (
            "You are a helpful summarizer. Produce a short bullet-point summary (3-6 bullets) of the following text chunk.\n"
            f"CHUNK #{i+1}:\n{c}\n\nSummary:" 
        )
        try:
            if client is None:
                # fallback: return first 600 chars of each chunk as "summary"
                chunk_summaries.append(c[:600])
                continue
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            s = resp.choices[0].message.content.strip()
            chunk_summaries.append(s)
        except Exception as e:
            chunk_summaries.append(f"[LLM failed to summarize chunk {i+1}: {e}]")

    # Reduce step: combine chunk summaries and synthesize
    combined = "\n\n".join(chunk_summaries)
    reduce_prompt = (
        "You are a synthesizer. Given the following bullet-point summaries from document chunks,"
        " produce a coherent high-level executive summary (approx 200-350 words). Avoid repetition.\n\n"
        f"INPUT:\n{combined}\n\nHigh-level summary:"
    )

    try:
        if client is None:
            # fallback: return combined
            return "[No OpenAI client] Combined chunk summaries:\n\n" + combined
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": reduce_prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        final_summary = resp.choices[0].message.content.strip()
        return f"**[Abstractive Summary]**\n\n{final_summary}"
    except Exception as e:
        return f"Failed reduce step: {e}\n\nCombined summaries:\n{combined}"

# -------------- Planner Agent ---------------
# The planner is a lightweight agent whose instruction is to choose one of the above tools
# based on user intent. The AgentSDK will route the call to the appropriate function_tool.

planner_system = (
    "You are a planner that decides which tool to call for a user's request.\n"
    "Available tools:\n"
    "- rag_tool(query, filename=None): retrieve passages and answer a specific question.\n"
    "- summarize_tool(filename=None): produce a high-level summary of a file or all files.\n\n"
    "Decision rules:\n"
    "1) If the user asks for 'summary', 'summarize', 'abstract', 'overview', or 'high-level', call summarize_tool.\n"
    "2) If the user asks a question about content (wh- words, 'what', 'who', 'when', 'how', 'why', 'explain', 'details', 'describe'), call rag_tool.\n"
    "3) If the user indicates a filename or says 'this file' try to pass filename where possible.\n"
    "4) If unsure, prefer rag_tool with a clarifying opening sentence.\n"
    "Return a short note stating which tool you called and then the tool output."
)

planner_agent = Agent(name="Planner", instructions=planner_system, tools=[rag_tool, summarize_tool])

# -------------- Runner wrapper ---------------

def run_planner(user_input: str) -> str:
    try:
        # Runner.run_sync will let the planner agent call rag_tool or summarize_tool as needed
        out = Runner.run_sync(planner_agent, input=user_input)
        # normalize output if it's a dict-like
        if hasattr(out, "output_text") and out.output_text:
            return out.output_text
        if isinstance(out, dict):
            for k in ("output_text", "output", "message", "result"):
                if k in out:
                    return out[k]
            return str(out)
        return str(out)
    except Exception as e:
        return f"Planner execution failed: {e}"

# -------------- Streamlit UI ---------------

st.sidebar.title("Upload & Process PDFs")
uploaded = st.sidebar.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])

if st.sidebar.button("Process uploaded PDFs"):
    if not uploaded:
        st.sidebar.warning("No files selected.")
    else:
        with st.spinner("Extracting text and indexing..."):
            for f in uploaded:
                name = getattr(f, "name", None) or f"uploaded_{len(st.session_state['files'])+1}.pdf"
                txt = extract_text_from_pdf_file(f)
                if not txt.strip():
                    st.sidebar.warning(f"No text extracted from {name} — it may be scanned or image-only PDF.")
                    continue
                st.session_state["files"][name] = txt
                chunks = chunk_text(txt)
                try:
                    vs = build_faiss_for_chunks(chunks)
                    # store metadata of chunks (optional) — FAISS from langchain stores metadata if provided
                    st.session_state["vectorstores"][name] = vs
                except Exception as e:
                    st.sidebar.error(f"Failed to build FAISS for {name}: {e}")
            st.sidebar.success("Processing complete.")

# File list + quick actions
st.sidebar.markdown("---")
st.sidebar.subheader("Processed files")
if st.session_state["files"]:
    for idx, fn in enumerate(st.session_state["files"].keys()):
        st.sidebar.write(f"{idx+1}. {fn}")
else:
    st.sidebar.info("No processed files yet.")

st.title("Multi-Agent PDF Assistant")
st.write("Upload PDFs in the sidebar and click 'Process uploaded PDFs'. Then ask questions or request summaries below.")

user_input = st.text_input("Ask a question or request a summary:")

if user_input:
    st.session_state["chat_history"].append(("user", user_input))
    with st.spinner("Planner deciding which agent to run..."):
        reply = run_planner(user_input)
    st.session_state["chat_history"].append(("agent", reply))

# Render chat
for role, msg in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
    else:
        st.markdown(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

st.caption("Planner agent decides between RAG (rag_tool) and Summarization (summarize_tool). Tools perform retrieval and LLM calls for grounded answers and abstractive summaries.")

# EOF
