# import os
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from api.pubmed_fetch import fetch_pubmed_articles  # Import PubMed function
# from dotenv import load_dotenv, find_dotenv
# import streamlit as st

# # Load environment variables
# load_dotenv(find_dotenv())

# # Load LLM (Mistral-7B)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#         task="text-generation",
#         temperature=0.5,
#         model_kwargs={"max_length": 512}
#     )
#     return llm

# # Load FAISS Database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# def get_pubmed_results(query, max_results=3):
#     """Fetch articles from PubMed and return formatted context."""
#     pubmed_articles = fetch_pubmed_articles(query, max_results=max_results)
#     if not pubmed_articles:
#         return "No relevant PubMed articles found."
#     return "\n\n".join([f"Title: {article['title']}\nAbstract: {article['abstract']}" for article in pubmed_articles])

# CUSTOM_PROMPT_TEMPLATE = """
# You are an AI medical assistant. Your task is to analyze the given context and provide a well-structured, clear, and accurate response.

# ### Guidelines:
# - **Think step by step** before answering.
# - **Summarize key points** from multiple sources instead of copy-pasting.
# - **Explain in simple, understandable language** while maintaining accuracy.
# - **Provide structured responses** (Definition, Symptoms, Diagnosis, Treatment, etc.).
# - **If you don't know the answer, say so. Don't make up facts.**

# ---

# ### **Context Information:**
# {context}

# ---

# ### **Question:**  
# {question}

# ---

# ### **Final Answer:**  
# (Present a clear, structured, and well-explained response based on the available context.)
# """

# def set_custom_prompt(custom_prompt_template):
#     return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# def hybrid_retrieval(user_query):
#     """Retrieves data from FAISS and PubMed, then combines them."""
#     # Retrieve from FAISS
#     faiss_docs = db.similarity_search(user_query, k=3)
#     faiss_context = "\n\n".join([doc.page_content for doc in faiss_docs])
    
#     # Retrieve from PubMed
#     pubmed_context = get_pubmed_results(user_query, max_results=3)
    
#     # Combine both sources
#     combined_context = f"FAISS Results:\n{faiss_context}\n\nPubMed Results:\n{pubmed_context}"
#     return combined_context

# # Create QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=load_llm(HUGGINGFACE_REPO_ID),
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 3}),
#     return_source_documents=True,
#     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
# )

# # # Get user query
# # user_query = input("Ask your medical question: ")
# # context = hybrid_retrieval(user_query)
# # response = qa_chain.invoke({'context': context, 'query': user_query})

# # print("\nRESULT:\n", response["result"])
# # print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
# # Streamlit UI
# st.title("Medical AI Chatbot")
# user_query = st.text_input("Ask your medical question:")

# if user_query:
#     with st.spinner("Searching medical sources..."):
#         context = hybrid_retrieval(user_query)
#         response = qa_chain.invoke({'context': context, 'query': user_query})

#     st.subheader("Response:")
#     st.write(response["result"])

#     st.subheader("Source Documents:")
#     for doc in response["source_documents"]:
#         st.write(doc.page_content)

import os
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from api.pubmed_fetch import fetch_pubmed_articles  # Import PubMed function
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Load LLM (Mistral-7B)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
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

# Custom Prompt for the LLM
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

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

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

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Medical AI Chatbot")

# Display Chat History (keeps past interactions visible)
st.subheader("Chat History:")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")  
    st.markdown(f"**Bot:** {chat['bot']}")  
    st.markdown("---")

# Input field with form submission
with st.form(key="chat_form"):
    user_query = st.text_input("Ask your medical question:", key="user_input")
    submit_button = st.form_submit_button("Submit")

if submit_button and user_query.strip():
    # Handle greetings separately
    greetings = ["hi", "hello", "hey", "whassup", "what's up", "how are you", "what are you doing"]
    
    if user_query.lower() in greetings:
        bot_response = "Hello! How can I assist you with medical information today? 😊"
    else:
        # Retrieve medical information
        with st.spinner("Searching medical sources..."):
            context = hybrid_retrieval(user_query)
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response = qa_chain.invoke({'context': context, 'query': user_query})
            bot_response = response["result"]

    # Store the interaction in session state
    st.session_state.chat_history.append({"user": user_query, "bot": bot_response})

    # Clear the input field after submission
    st.session_state.user_input = ""
    st.rerun()  # Refresh UI to update chat history

