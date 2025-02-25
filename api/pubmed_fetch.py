import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Your PubMed API key
NCBI_API_KEY = os.environ.get("PUBMED_API_KEY")

def fetch_pubmed_articles(query, max_results=5):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    # Searching PubMed
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json&api_key={NCBI_API_KEY}"
    search_results = requests.get(search_url).json()
    article_ids = search_results["esearchresult"]["idlist"]

    if not article_ids:
        return [{"title": "No articles found", "abstract": "Try a different query."}]

    # Fetch article details
    fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(article_ids)}&retmode=xml&api_key={NCBI_API_KEY}"
    fetch_results = requests.get(fetch_url)
    
    soup = BeautifulSoup(fetch_results.content, "xml")
    articles = []
    
    for article in soup.find_all("PubmedArticle"):
        title = article.find("ArticleTitle").text
        abstract = article.find("AbstractText")
        abstract_text = abstract.text if abstract else "No abstract available."
        
        articles.append({"title": title, "abstract": abstract_text})

    return articles