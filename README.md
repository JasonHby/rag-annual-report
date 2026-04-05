# Financial Report RAG Assistant

A simple end-to-end RAG system for question answering over corporate annual reports using LangChain, Chroma, FastAPI, and Groq.

## Overview

This project demonstrates a basic retrieval-augmented generation pipeline over a PDF annual report. It loads report content, splits it into searchable chunks, stores embeddings in a local vector database, retrieves relevant evidence for a user query, and sends grounded context to an LLM to generate an answer with cited sources.

## Tech Stack

- Python
- LangChain
- Chroma
- HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)
- FastAPI
- Groq

## Project Structure

```text
.
├── day1_chunking.py
├── day2_retrieval.py
├── day3_api.py
├── README.md
└── .gitignore

Core Functions
PDF Ingestion and Chunking
The report is loaded with PyPDFLoader and split into smaller overlapping chunks using RecursiveCharacterTextSplitter. This makes the document easier to index and retrieve from later.

Embeddings and Retrieval
Each chunk is converted into a vector using HuggingFaceEmbeddings and stored locally in Chroma. When a user asks a question, the query is embedded and matched against stored chunk vectors to retrieve the most relevant content.

API and Answer Generation
A FastAPI endpoint accepts a question, retrieves the top relevant chunks, formats them into prompt context, and sends that context to a Groq-hosted LLM. The API returns both the generated answer and the supporting source chunks with page references.

Setup
Create and activate a virtual environment, then install the dependencies.

pip install langchain-community langchain-text-splitters pypdf
pip install langchain-huggingface langchain-chroma chromadb sentence-transformers
pip install fastapi uvicorn langchain-openai
Environment Variable
Set your Groq API key before running the API.

PowerShell
$env:GROQ_API_KEY="your_api_key"
Bash
export GROQ_API_KEY="your_api_key"
Usage
Run PDF chunking
python day1_chunking.py
Build the vector database and test retrieval
python day2_retrieval.py
Start the API server
uvicorn day3_api:app --reload
Then open:

http://127.0.0.1:8000/docs
Example Query
Example request body in /docs:

{
  "question": "What were the company's total sales in 2025?"
}
Example response:

{
  "answer": "Total sales revenue for 2025: $4,821,851 (in thousands).",
  "sources": [
    {
      "source": "annual_report.pdf",
      "page": 55,
      "page_label": "56",
      "score": 0.40,
      "content": "Selected Annual Information ..."
    }
  ]
}
Notes
The PDF file is not included in this repository.
Place the annual report PDF in the project root before running the scripts.
Retrieval quality depends on PDF text quality and query phrasing.
This is a lightweight demo project intended for learning and interview preparation.
Possible Improvements
Improve PDF text cleaning for tables, headers, and repeated formatting noise
Add hybrid retrieval or reranking for better precision on financial questions
Introduce query rewriting to improve retrieval quality
Add a lightweight frontend on top of the FastAPI backend
Generalize the pipeline to support multiple reports instead of a single PDF