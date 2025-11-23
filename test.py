
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

def extract_text_from_url(url: str) -> str:
    """Extract all visible text from a webpage using Selenium (not headless)."""
    options = Options()
    # Not headless, browser will be visible
    options.add_argument("--start-maximized")
    
    service = Service()  # Make sure chromedriver is in PATH
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.get(url)
    time.sleep(5)  # Wait for page to load completely (adjust if needed)
    
    # Optionally scroll down to load dynamic content
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    
    # Combine all paragraph texts
    text = "\n".join([p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()])
    return text

def chunk_text(text: str, chunk_size: int = 300):
    """Split text into chunks of approximately `chunk_size` words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks, model_name='paraphrase-multilingual-mpnet-base-v2'):
    """Convert text chunks into vectors using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    vectors = [model.encode(chunk) for chunk in chunks]
    return np.array(vectors).astype('float32'), model

def search_vectors(query, chunks, vectors, model, top_k=3):
    """Search query against vectors and return top-k chunks."""
    query_vec = model.encode([query]).astype('float32')
    similarities = cosine_similarity(query_vec, vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "similarity_score": float(similarities[idx])
        })
    return results


if __name__ == "__main__":
    url = "https://www.digikala.com/"
    
    # Step 1: Extract website text using Selenium (visible browser)
    full_text = extract_text_from_url(url)
    print(f"\nExtracted {len(full_text.split())} words from the website.\n")
    
    # Show first 1000 characters of extracted text
    print("----- Extracted Text (Preview) -----")
    print(full_text[:1000] + "\n...")
    
    # Step 2: Split into chunks
    chunks = chunk_text(full_text, chunk_size=200)
    print(f"\nSplit text into {len(chunks)} chunks.\n")
    
    # Step 3: Embed chunks
    vectors, model = embed_chunks(chunks)
    print(f"Embedded chunks into vectors of dimension {vectors.shape[1]}.\n")
    
    # Step 4: Search
    query = "تخفیف"
    top_k = 3
    results = search_vectors(query, chunks, vectors, model, top_k=top_k)
    
    # Step 5: Show results
    print("\n----- Top Results -----")
    for r in results:
        print(f"Score: {r['similarity_score']:.4f}\nChunk: {r['chunk']}\n---")
