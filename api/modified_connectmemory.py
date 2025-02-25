import os

from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pubmed_fetch import fetch_pubmed_articles
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Calls API from .env
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Loads model from Hugging Face
print("HF_TOKEN successfully loaded.")  # Debugging step

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,  # Explicitly passing token
        task="text-generation",  # Explicitly specify task
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )
    return llm

# Step 2: Fetch data from PubMed API
def get_pubmed_results(query, max_results=3):
    """
    Fetch articles from PubMed and return formatted context.
    """
    pubmed_articles = fetch_pubmed_articles(query, max_results=max_results)
    
    if not pubmed_articles:
        return "No relevant PubMed articles found."

    # Format PubMed results as a string
    context = "\n\n".join([f"Title: {article['title']}\nAbstract: {article['abstract']}" for article in pubmed_articles])
    
    return context

# Step 3: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Now invoke with a single query
user_query = input("Write Query Here: ")

# Get context from PubMed only
context = get_pubmed_results(user_query, max_results=3)

# Pass the context to LLM
llm = load_llm(HUGGINGFACE_REPO_ID)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

response = qa_chain.invoke({'query': user_query, 'context': context})

print("\nRESULT:\n", response["result"])