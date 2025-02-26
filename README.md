
# **Medical Chatbot with Retrieval-Augmented Generation (RAG) and PubMed API**  

This project is a **medical chatbot** that retrieves accurate, context-aware medical information using **Retrieval-Augmented Generation (RAG)**. It combines **LangChain, Hugging Face's Mistral-7B, FAISS vector search, and the PubMed API** to provide structured and reliable medical responses.  

## **Features**  

- **Retrieval-Augmented Generation (RAG):** Enhances AI-generated responses with document retrieval.  
- **FAISS Vector Database:** Efficiently stores and retrieves medical text embeddings.  
- **PubMed API Integration:** Fetches real-time medical research articles.  
- **Custom Prompt Engineering:** Improves response structure and clarity.  
- **Streamlit-Based UI:** Provides an interactive chatbot interface.  
- **Multi-PDF Support:** Extracts knowledge from multiple research papers.  

## **Project Structure**  

```
📁 MedicalChatbot
│   ├── .env                      # Stores API keys
│   ├── .gitignore                # Ignores unnecessary files (venv, __pycache__, etc.)
│   ├── Hybrid_ChatBot.py         # Main chatbot script
│   ├── README.md                 # Documentation
│   ├── requirements.txt          # Project dependencies
│
├── 📂 api
│   ├── medibot.py                # Chatbot API logic
│   ├── modified_connectmemory.py  # Alternative FAISS connection script
│   ├── modified_medibot.py       # Modified chatbot logic
│   ├── pubmed_fetch.py           # Fetches real-time research from PubMed
│   └── __pycache__/              # Compiled Python files
│
├── 📂 FAISS_connect
│   ├── connect_memory_with_llm.py  # Connects FAISS with LLM
│   ├── create_memory_for_llm.py    # Loads PDFs & creates FAISS vector store
│   ├── medibot.py                  # Chatbot logic using FAISS
│   ├── pdf_test.py                 # PDF processing test script
│   └── data/                        # Medical PDFs for knowledge base
│       ├── Multiple medical research papers
│
└── 📂 venv/                          # Virtual environment for dependencies
```

## **Installation**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/vinesh1054/MedicalChatBot01.git
cd MedicalChatBot01
```

### **2. Set Up Virtual Environment**  
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4. Set Up API Keys**  
- Create a `.env` file in the project root and add the following:  
```sh
HF_TOKEN=your_huggingface_api_key
PUBMED_API_KEY=your_pubmed_api_key
```

## **Usage**  

### **1. Process Medical PDFs and Create FAISS Database**  
```sh
python FAISS_connect/create_memory_for_llm.py
```

### **2. Start the Chatbot**  
```sh
streamlit run api/medibot.py
```

## **How It Works**  

1. **Document Ingestion:** Extracts text from multiple medical research papers.  
2. **Vector Storage:** Stores text embeddings using FAISS for efficient retrieval.  
3. **Information Retrieval:** Searches FAISS for relevant medical context.  
4. **PubMed API Integration:** Fetches real-time medical research.  
5. **AI Response Generation:** Uses Mistral-7B to generate structured responses.  
6. **Interactive Chatbot UI:** Provides a user-friendly chatbot interface.  

## **PubMed API Integration**  

This chatbot integrates the **PubMed API** to fetch recent and relevant medical research articles.  

### **Example API Usage**  
```python
from api.pubmed_fetch import search_pubmed, fetch_pubmed_details

query = "diabetes treatment"
pmids = search_pubmed(query)
articles = fetch_pubmed_details(pmids)

for article in articles:
    print(f"Title: {article['title']}")
    print(f"Authors: {article['authors']}")
    print(f"Source: {article['source']}")
    print(f"Read More: {article['link']}\n")
```

## **Future Enhancements**  

- Fine-tune **Mistral-7B** on medical datasets for better accuracy.  
- Implement **speech-to-text and text-to-speech** for voice interaction.  
- Deploy on **Hugging Face Spaces or AWS**.  
- Introduce **hybrid search (BM25 + FAISS)** for improved document retrieval.  

## **References**  

- [LangChain Documentation](https://python.langchain.com/)  
- [FAISS Documentation](https://faiss.ai/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [PubMed API Guide](https://www.ncbi.nlm.nih.gov/home/develop/api/)  
- [Streamlit Documentation](https://streamlit.io/)  

