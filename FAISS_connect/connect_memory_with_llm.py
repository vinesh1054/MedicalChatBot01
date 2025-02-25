import os

from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub



## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace) 
HF_TOKEN=os.environ.get("HF_TOKEN") #calls API from .env
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3" #loads whole Huggingface repo of this model
print("HF_TOKEN successfully loaded.")  # Debugging step

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),  # Explicitly passing token
        task="text-generation",  # Explicitly specify task
        temperature=0.5,
        model_kwargs={
            "max_length": 512
        }
        
    )
    return llm






# Step 2: Connect LLM with FAISS and Create chain

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

### **Your Thought Process:**  
1. What key information is available from the context?  
2. How do these pieces of information relate to the question?  
3. What is the most accurate and complete way to answer?  

---

### **Final Answer:**  
(Present a clear, structured, and well-explained response based on the above analysis.)
"""




def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])