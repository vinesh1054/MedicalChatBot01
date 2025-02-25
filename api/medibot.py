import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

from pubmed_fetch import fetch_pubmed_articles  # Import PubMed fetch function

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

HF_TOKEN = os.getenv("HF_TOKEN")  
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" 

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,  
        task="text-generation",
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )
    return llm

def is_greeting(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "greetings"]
    return user_input.lower() in greetings

def main():
    st.title("AI Medical Chatbot with PubMed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Enter your medical query:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Check for greetings and respond without searching PubMed or FAISS
        if is_greeting(prompt):
            greeting_response = "Hello! How can I assist you with your medical queries today?"
            st.chat_message('assistant').markdown(greeting_response)
            st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
            return

        try:
            # 🔹 Step 1: Fetch data from PubMed
            articles = fetch_pubmed_articles(prompt)

            # 🔹 Step 2: If PubMed has results, display them
            if articles and articles[0]["title"] != "No articles found":
                result_to_show = "**🔍 PubMed Results:**\n\n"
                for i, article in enumerate(articles):
                    result_to_show += f"**{i+1}. {article['title']}**\n{article['abstract']}\n\n"
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
                return  # Skip FAISS search if PubMed has data

            # 🔹 Step 3: If PubMed fails, fallback to FAISS
            st.write("No relevant PubMed articles found. Searching FAISS...")
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load FAISS vector store")

            CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you don’t know the answer, just say that you don’t know. 
                Don’t provide anything outside the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
            """

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            result_to_show = result + "\n\n**Source Docs:**\n" + str(source_documents)

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
