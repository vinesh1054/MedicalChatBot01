

import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from api.pubmed_fetch import fetch_pubmed_articles  # Import PubMed function
from dotenv import load_dotenv, find_dotenv
import streamlit as st

# Load environment variables
load_dotenv(find_dotenv())

# Load LLM (Mistral-7B)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """Loads the Hugging Face model."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )
    return llm

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def get_pubmed_results(query, max_results=3):
    """Fetch articles from PubMed and return formatted context."""
    pubmed_articles = fetch_pubmed_articles(query, max_results=max_results)
    if not pubmed_articles:
        return "No relevant PubMed articles found."
    return "\n\n".join([f"Title: {article['title']}\nAbstract: {article['abstract']}" for article in pubmed_articles])

CUSTOM_PROMPT_TEMPLATE = """
You are an AI medical assistant. Your task is to analyze the given context and provide a well-structured, clear, and accurate response.

### Guidelines:
- **Think step by step** before answering.
- **Summarize key points** from multiple sources instead of copy-pasting.
- **Explain in simple, understandable language** while maintaining accuracy.
- **Provide structured responses** (Definition, Symptoms, Diagnosis, Treatment, etc.).
- **If you don't know the answer, say so. Don't make up facts.**

---

### **Context Information:**
{context}

---

### **Question:**  
{question}

---

### **Final Answer:**  
(Present a clear, structured, and well-explained response based on the available context.)
"""

def hybrid_retrieval(user_query):
    """Retrieves data from FAISS and PubMed, then combines them."""
    # Retrieve from FAISS
    faiss_docs = db.similarity_search(user_query, k=3)
    faiss_context = "\n\n".join([doc.page_content for doc in faiss_docs])
    
    # Retrieve from PubMed
    pubmed_context = get_pubmed_results(user_query, max_results=3)
    
    # Combine both sources
    combined_context = f"FAISS Results:\n{faiss_context}\n\nPubMed Results:\n{pubmed_context}"
    return combined_context

def answer_query(user_query):
    """Generates an answer using both FAISS and PubMed results."""
    context = hybrid_retrieval(user_query)  # Get FAISS + PubMed results
    llm = load_llm(HUGGINGFACE_REPO_ID)  # Load model
    response = llm.invoke(CUSTOM_PROMPT_TEMPLATE.format(context=context, question=user_query))
    return response

# Streamlit UI
st.title("Medical AI Chatbot")

user_query = st.text_input("Ask your medical question:")

if user_query:
    with st.spinner("Searching medical sources..."):
        response = answer_query(user_query)

    st.subheader("Response:")
    st.write(response)

    # Optionally, show the retrieved context
    st.subheader("Retrieved Context:")
    st.write(hybrid_retrieval(user_query))
