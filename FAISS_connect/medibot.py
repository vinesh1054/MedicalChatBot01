import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss" #location of my database
@st.cache_resource #storing database as cache
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

HF_TOKEN = os.getenv("HF_TOKEN")  
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" 


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,  
        task="text-generation",  # Explicitly specify task
        temperature=0.5,
        model_kwargs={
            "max_length": 512
        }
    )
    return llm




def main():
    st.title("Ask Question")

        # Step 1: Initialize chat history, #streamlit refreshs and doesnt store previous chats.
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Create empty list for storing messages, #this is to store chats

    # Step 2: Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    prompt = st.chat_input("Enter your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt) #input to bot by user
        st.session_state.messages.append({'role':'user', 'content': prompt}) #Store user's message

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore() #vector store is loaded now 
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt}) #this is the prompt given by the user

            result=response["result"]
            source_documents=response["source_documents"] #to print source documenrts
            result_to_show=result+"\n \n Source Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

        #to check if streamlit is working correctly:---
        # response = "Hi, I am Medibot"
        # st.chat_message("assistent").markdown(response) #response by bot, here its fixed since i specidfied
        # st.session_state.messages.append({"role": "assistant", "content": response}) #Store AI response

    


if __name__ == "__main__":
    main()

    

