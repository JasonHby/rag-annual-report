from day1_chunking import get_document_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

PDF_FILE = "martinrea_report.pdf"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def build_vector_database():
    chunks = get_document_chunks(PDF_FILE)
    print(f"Total chunks created: {len(chunks)}")
    
    # Initialize Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    # Create Chroma Vector Database
    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = CHROMA_DB_DIR,
    )
    print("Vector database created and persisted successfully.")
    return vector_store

def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME
    )
    vector_store = Chroma(
        persist_directory = CHROMA_DB_DIR,
        embedding_function = embeddings,
    )
    return vector_store

def search_database(query, top_k):
    vector_store = get_or_create_vector_database()
    # similarity search process
    # 1. do vector embedding for the query
    # 2. calculate similarity score between query embedding and document embeddings
    # 3. return top_k most similar documents
    docs = vector_store.similarity_search_with_score(query, k = top_k)
    return docs

def get_or_create_vector_database():
    db_path = Path(CHROMA_DB_DIR)
    if db_path.exists() and any(db_path.iterdir()):
        print("Loading existing vector database...")
        return load_vector_database()
    else:
        print("No existing vector database found. Building a new one...")
        return build_vector_database()

if __name__ == "__main__":
    # Step 1: Build and persist the vector database
    # build_vector_database()

    # Step 2: Mock a search query
    query = "total sales revenue 2025 consolidated statements operations"
    results = search_database(query, top_k = 5)
    print(f"\nUser Query: {query}")
    print(f"\nDetetced {len(results)} relevant chunks\n")
    for i,(doc,score) in enumerate(results):
        clean_metadata = {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "page_label": doc.metadata.get("page_label"),
        }
        
        print(f"==== Top {i+1} Chunk ====")
        print(f"Similarity Score: {score}")
        print("Content:")
        print(doc.page_content[:1000])
        print("\nMetadata:")
        print(clean_metadata)
        print("\n")


    