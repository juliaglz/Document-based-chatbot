# 📚 Modular Document-Based Chatbot

This project is a modular, document-based chatbot that allows flexible integration of different components for text parsing, chunking, embedding, and response generation using multiple LLMs. It is built for scalability, experimentation, and easy testing of various NLP techniques.

---

## 🚀 Features

- **Modular Architecture** – Easily swap components to experiment with different techniques.
- **Multi-format Parser** – Reads text from `.pdf`, `.docx`, `.pptx`, and `.xlsx` files.
- **Intelligent Text Chunking** – Splits documents at natural sentence endings.
- **Multiple Embedding Methods**:
  - TF-IDF (cosine similarity)
  - BERT embeddings
  - ChromaDB native embeddings
- **Vector Database Integration** – Uses [ChromaDB](https://www.trychroma.com/) for fast semantic search.
- **LLM Response Generation** – Supports:
  - GPT-2
  - T5 (Text-to-Text Transfer Transformer)
  - GPT-3.5 (via Azure OpenAI)
- **Streamlit UI** – Simple web-based interface to interact with the chatbot.

---

## 🧩 Architecture Overview

[Parser] → [Chunker] → [Embedder (TFIDF | BERT | Chroma)] → [ChromaDB] → [LLM (GPT2 | T5 | GPT3.5)]

### 1. Parser

Reads text from documents using:
- `PyPDF2` for PDFs
- `python-docx` for Word files
- `python-pptx` for PowerPoint
- `openpyxl` for Excel

### 2. Chunker

Splits long text into logical chunks by identifying sentence boundaries (`.`, `?`, `!`).

### 3. Embedder

Transforms chunks into vector representations using:
- **TF-IDF** with cosine similarity
- **BERT embeddings**
- **ChromaDB's built-in embeddings**

### 4. Vector Database

ChromaDB efficiently stores and queries embeddings for semantic similarity.

### 5. LLMs

Generates context-aware answers using:
- Open-source models: `GPT-2`, `T5`
- Azure-hosted: `GPT-3.5`

---

## 🖥️ UI

The chatbot runs on **Streamlit** and offers a basic interface to:
- Upload documents
- Ask questions
- View references and generated answers

---

## 📦 Dependencies

- `transformers`
- `scikit-learn`
- `sentence-transformers`
- `chromadb`
- `openai` (Azure)
- `PyPDF2`, `python-docx`, `python-pptx`, `openpyxl`
- `streamlit`
- `python-dotenv`
