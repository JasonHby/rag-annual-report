# Financial Report RAG Assistant

A simple end-to-end RAG system for question answering over corporate annual reports using LangChain, Chroma, FastAPI, and Groq.

## Overview

This project demonstrates a basic retrieval-augmented generation pipeline over a PDF annual report:

1. Load and chunk the report into searchable text segments
2. Generate embeddings and store them in a local Chroma vector database
3. Retrieve the most relevant chunks for a user query
4. Send grounded context to an LLM and return both the answer and cited sources

## Tech Stack

- Python
- LangChain
- Chroma
- HuggingFace Embeddings (`BAAI/bge-small-en-v1.5`)
- FastAPI
- Groq

## Project Structure

- `day1_chunking.py`: PDF ingestion and text chunking
- `day2_retrieval.py`: local embeddings, Chroma storage, and retrieval
- `day3_api.py`: FastAPI endpoint and LLM-based answer generation

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
pip install langchain-community langchain-text-splitters pypdf
pip install langchain-huggingface langchain-chroma chromadb sentence-transformers
pip install fastapi uvicorn langchain-openai

Environment Variable
Set your Groq API key before running the API:

export GROQ_API_KEY="your_api_key"
On PowerShell:

$env:GROQ_API_KEY="your_api_key"
Run
1. Chunk the document
python day1_chunking.py
2. Build and test retrieval
python day2_retrieval.py
3. Start the API server
uvicorn day3_api:app --reload
Then open:

http://127.0.0.1:8000/docs
Example Query
{
  "question": "What were the company's total sales in 2025?"
}
Notes
The PDF file is not included in this repository.
Place the annual report PDF in the project root as martinrea_report.pdf.
Retrieval quality depends on PDF text quality and query phrasing.
