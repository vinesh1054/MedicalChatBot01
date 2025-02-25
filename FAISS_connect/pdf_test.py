from langchain_community.document_loaders import PyPDFLoader

pdf_path = "FAISS_connect\data\Robbins & Cotran Pathologic Basis of Disease (Robbins Pathology).pdf"  # Replace with an actual file from your folder

print("Loading a single PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages.")
