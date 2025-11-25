This project contains **two different implementations** of a **multi-agent PDF question-answering and summarization system**.
Both systems perform **PDF ingestion, chunking, embedding, retrieval, summarization**, and a **planner agent that automatically selects the right agent**.

However, the two implementations differ greatly in their **architecture**, **dependencies**, and **execution mode**.

---

# 🔵 **Code Version 1 – Online Multi-Agent RAG Using AgentSDK + OpenAI API - planagent.py**

### 📌 **Concept**

This version uses the **OpenAI AgentSDK**, a full multi-agent architecture with:

* A **Planner Agent**
* A **RAG Tool Agent**
* A **Summarization Tool Agent**
* A **Runner** that allows the planner to call tools dynamically
* Online LLM calls using **OpenAI GPT-4o-mini**

It requires an **active internet connection** and a **valid OpenAI API key**.

---

## 🔧 **Key Components**

### **1. RAG Tool (rag_tool)**

* Works as a tool registered inside the agents system.
* Searches inside FAISS vectorstores created from uploaded PDFs.
* Retrieves top-k chunks and sends them to the LLM.
* The LLM generates an answer **fully grounded in the retrieved context**.

### **2. Summarization Tool (summarize_tool)**

* Also a tool registered inside the agent system.
* Performs **map-reduce summarization**:

  * Chunk-level summaries
  * Final combined executive summary

### **3. Planner Agent (AgentSDK)**

* Reads user requests and decides:

  * *If request is a question → call rag_tool*
  * *If request asks for summary → call summarize_tool*
* The planner **does not answer**, it only decides.
* The Runner executes the tool call.

### **4. Online LLM Dependency**

Uses:

```python
client = OpenAI(api_key=...)
```

and sends:

```python
client.chat.completions.create(...)
```

So this version is **cloud-dependent**.

---

## 📊 **Use Case**

* Best for **high-quality LLM answers**
* When **internet** and **OpenAI API quota** are available
* For teams implementing **advanced multi-agent architectures**
* Provides **tool execution traceability** and **planner reasoning**

---

---

# 🟢 **Code Version 2 – Fully Offline Multi-Agent RAG (No API Calls) - planoffagent.py**

### 📌 **Concept**

This version simulates a multi-agent system **without any API calls**.

All intelligence is local:

* Embeddings → `sentence-transformers`
* Summarization → `facebook/bart-large-cnn`
* Planner agent → simple Python class
* No OpenAI and no cloud usage

Runs entirely **offline**, provided your machine supports PyTorch.

---

## 🔧 **Key Components**

### **1. RAG Agent (Offline CPU/GPU)**

* Searches FAISS for top-k similar chunks.
* Summarizes retrieved chunks using the local BART model.
* Produces a final condensed answer.

### **2. Summarization Agent**

* Summarizes the **full document** using the local BART summarizer.
* Also uses **multi-chunk map-reduce**, but done entirely offline.

### **3. Planner Agent (Rule-Based)**

Simple keyword rules:

* If message contains “summary”, “overview”, “abstract” → use SummarizationAgent
* Otherwise → use RAGAgent

### **4. No LLM API**

This version does **not** use:

* OpenAI API
* AgentSDK
* Cloud models

Everything runs on:

```python
pipeline("summarization", model="facebook/bart-large-cnn")
```

---

## 📊 **Use Case**

* Ideal when **internet is unavailable**
* Avoids quota issues (e.g. 429 insufficient_quota error)
* Suitable for secure/lab environments
* Runs slower but fully local

---

---

# ⚖️ **Comparison Table**

| Feature       | Code Version 1 (Online AgentSDK)     | Code Version 2 (Offline)                   |
| ------------- | ------------------------------------ | ------------------------------------------ |
| Execution     | Cloud (OpenAI)                       | Fully Local                                |
| Planner       | AgentSDK multi-agent planner         | Simple rule-based class                    |
| Agents        | Tools registered via `function_tool` | Python classes acting as agents            |
| Summarization | GPT-4o-mini (LLM)                    | BART offline summarizer                    |
| RAG Answer    | GPT-based answer synthesis           | Local summarizer compresses retrieved text |
| Accuracy      | Much higher                          | Limited by offline models                  |
| Dependencies  | AgentSDK + OpenAI client             | HuggingFace Transformers only              |
| PDFs          | ✔️                                   | ✔️                                         |
| FAISS         | ✔️                                   | ✔️                                         |
| Cost          | Requires API usage                   | Free                                       |
| Speed         | Fast, cloud accelerated              | Slower on CPU                              |

---


---

## 📘 Project Overview

This repository contains **two independent implementations** of a **multi-agent PDF retrieval and summarization system**:

### **1. Online Multi-Agent System (AgentSDK + OpenAI)**

A cloud-powered system using:

* OpenAI AgentSDK
* Planner agent
* RAG tool agent
* Summarization tool agent
* GPT-4o-mini for high-quality answers

This version provides **true autonomous tool-calling agents**, where the planner decides whether to retrieve content or summarize based on the user's intent.

---

### **2. Offline Multi-Agent System (No API Needed)**

A completely local version using:

* HuggingFace Transformers (`bart-large-cnn`)
* Sentence-transformers embeddings
* FAISS retrieval
* A simple rule-based planner

This version does **not require any API key**, works **100% offline**, and is suitable for secure environments or machines without internet access.

---

## 🧠 Multi-Agent Workflow

Both systems follow the same conceptual flow:

1. **PDF → Text extraction**
2. **Chunking**
3. **Embeddings + FAISS index**
4. User asks:

   * A **question** → → RAG agent retrieves + answers
   * A **summary request** → → Summarization agent
5. **Planner** selects the right agent automatically


