# 📚 Multi-Document Conversational RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system built with Streamlit, Google Gemini API, and NumPy. The assistant indexes multiple PDF documents, performs vector similarity search, and answers user queries with exact source page citations and multi-turn conversational memory.

## 🚀 Key Features
- **Multi-Document Ingestion:** Processes multiple PDFs simultaneously using PyPDF.
- **Sliding Window Chunking:** 700-character chunks with a 120-character overlap to preserve semantic context across page boundaries.
- **Vector Search:** Computes cosine similarity across high-dimensional embeddings using NumPy.
- **Conversational Memory & Query Condensation:** Reformulates follow-up queries into standalone search queries using Gemini before retrieval.
- **Source Attribution:** Interactive dropdown citing document names, page numbers, and exact excerpts.
- **Configurable Retrieval (Top-K):** Real-time sidebar slider to adjust retrieved context size.

## 🛠️ Tech Stack
- **Framework:** Streamlit
- **LLM & Embeddings:** Google Gemini API (embedding-001 & gemini-3.8-flash)
- **PDF Extraction:** PyPDF
- **Vector Operations:** NumPy

## ⚙️ Local Setup

1. **Clone repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/my-first-rag.git](https://github.com/YOUR_USERNAME/my-first-rag.git)
   cd my-first-rag