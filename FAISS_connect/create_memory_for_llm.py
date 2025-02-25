

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Loading raw PDF(s)
DATA_PATH = "FAISS_connect\data"  # Path to the folder containing PDFs

def load_pdf_files(data):
    """Loads PDFs from a directory and handles potential errors."""
    try:
        loader = DirectoryLoader(
            data,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,  # Show progress bar while loading
            use_multithreading=True  # Load PDFs faster
        )
        # loader = PyPDFLoader("FAISS_connect\data\Guyton_and_Hall_Textbook_of_Medical_Physiology_14th_Ed.pdf")
        print("Starting PDF loading...")
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDF files.")
        return documents
    except Exception as e:
        print(f" Error loading PDFs: {e}")
        return []

documents = load_pdf_files(DATA_PATH)

# Step 2: Create Chunks
def create_chunks(extracted_data, chunk_size=1000, chunk_overlap=100):
    """Splits extracted text into overlapping chunks for better context retention."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    """Loads HuggingFace embeddings model for text vectorization."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Enable FAISS HNSW indexing for faster retrieval
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print(f" FAISS vector database saved at {DB_FAISS_PATH}.")


