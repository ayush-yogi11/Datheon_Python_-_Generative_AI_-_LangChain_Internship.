# Autonomous AI Agents & Generative AI Workshop Tasks

A comprehensive repository showcasing production-ready implementations of **Structured Data Extraction**, **Document RAG with Vector Embeddings**, **Multi-Tool Autonomous Agents**, and an **Interactive Full-Stack Streamlit Chatbot** powered by Python, Google Gemini, and LangChain.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Tasks & Architectures](#-tasks--architectures)
  - [Task 1: Structured LLM Prompt Pipeline & JSON Extractor](#task-1-structured-llm-prompt-pipeline--json-extractor)
  - [Task 2: Document Q&A Assistant with RAG & Vector Embeddings](#task-2-document-qa-assistant-with-rag--vector-embeddings)
  - [Task 3: Multi-Tool Autonomous AI Agent](#task-3-multi-tool-autonomous-ai-agent)
  - [Task 4: Interactive Full-Stack AI Chatbot with Streamlit](#task-4-interactive-full-stack-ai-chatbot-with-streamlit)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Running the Applications](#-running-the-applications)

---

## 💡 Project Overview
This repository contains a step-by-step modular progression from basic structured LLM output generation to complex multi-tool execution pipelines and interactive web interfaces:

1. **Schema Validation:** Enforcing structured JSON data outputs from unstructured text using Pydantic schemas.
2. **Retrieval-Augmented Generation (RAG):** Context-aware document Q&A backed by vector databases and semantic search.
3. **Autonomous Tool-Calling Agents:** ReAct-style agent loops capable of dynamically using web search, calculators, and file system utilities.
4. **Interactive UI Deployment:** Full-stack browser chat experience with real-time streaming and conversational memory management.

---

## 🛠️ Tasks & Architectures

### Task 1: Structured LLM Prompt Pipeline & JSON Extractor
* **Goal:** Extract structured data from unstructured text using LLM APIs and Python.
* **Key Features:**
  * Strict schema enforcement using `pydantic` models.
  * Integration with the modern `google-genai` SDK (`gemini-2.5-flash`).
  * Parameterized prompt templates ensuring low-temperature, deterministic data extraction.
* **Deliverable:** `task1/extractor.py`

### Task 2: Document Q&A Assistant with RAG & Vector Embeddings
* **Goal:** Build a RAG system for context-grounded question answering on custom documents with source citations.
* **Key Features:**
  * Recursive text splitting (`RecursiveCharacterTextSplitter`) for optimal chunk sizes.
  * Local high-performance vector embeddings via `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
  * Vector persistence using **ChromaDB**.
  * Grounded answer generation with Gemini and strict source transparency.
* **Deliverable:** `task2/rag_assistant.py`

### Task 3: Multi-Tool Autonomous AI Agent
* **Goal:** Create an autonomous AI agent capable of invoking tools to solve multi-step, compound research problems.
* **Key Features:**
  * Autonomous reasoning loop displaying Thought $\rightarrow$ Action $\rightarrow$ Observation reasoning steps.
  * Integrated custom tools:
    * **Web Search Tool:** Real-time web search powered by DuckDuckGo (`ddgs`).
    * **Math Calculator Tool:** Evaluation of mathematical operations.
    * **Local File Reader Tool:** Reading workspace files and documents dynamically.
* **Deliverable:** `task3/agent.py`

### Task 4: Interactive Full-Stack AI Chatbot with Streamlit
* **Goal:** Deploy a responsive web app with conversation memory and streaming AI responses.
* **Key Features:**
  * Streamlit web interface (`st.chat_message`, `st.chat_input`).
  * State management utilizing `st.session_state` across multi-turn user sessions.
  * Real-time token streaming direct from Google Gemini.
* **Deliverable:** `task4/app.py`

---

## 📁 Repository Structure

```text
ai-agentic-workshop/
├── .env                  # Environment variables configuration (API keys)
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
├── data/                 # Data directory containing sample documents/PDFs
│   └── company_policy.txt
├── task1/                # Task 1: Structured Prompt Pipeline & Extractor
│   └── extractor.py
├── task2/                # Task 2: RAG Pipeline with ChromaDB Vector Store
│   ├── rag_assistant.py
│   └── chroma_db/        # Local persistent vector database
├── task3/                # Task 3: Autonomous Multi-Tool Agent
│   └── agent.py
└── task4/                # Task 4: Full-Stack Streamlit Web Chatbot
    └── app.py
