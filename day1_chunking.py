from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_document_chunks(file_path):
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found:{pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
    )

    chunks = splitter.split_documents(pages)
    return chunks

if __name__ == "__main__":
    pdf_file = "martinrea_report.pdf"
    chunks = get_document_chunks(pdf_file)
    print(f" Total chunks created: {len(chunks)}")
    if chunks:
        print("\n==== Chunk 0 content ====\n")
        print(chunks[0].page_content[:500])

        print("\n==== Chunk 0 metadata ====\n")
        print(chunks[0].metadata)
    else:
        print("No chunks created.")