import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Load your OPENAI_API_KEY from .env
load_dotenv()
client = OpenAI()

# --- 1. Load PDF Text ---
print("--> Step 1: Reading document.pdf...")
reader = PdfReader("document.pdf")
raw_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + "\n"

# --- 2. Chunking ---
print("--> Step 2: Splitting into chunks...")
# Split into paragraphs or small ~400 character slices
paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30]
print(f"Created {len(paragraphs)} text chunks.")

# --- 3. Compute Embeddings ---
print("--> Step 3: Generating embeddings for chunks...")
def get_embedding(text):
    res = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return res.data[0].embedding

chunk_embeddings = [get_embedding(chunk) for chunk in paragraphs]

# --- 4. Question & Retrieval via Cosine Similarity ---
query = "What is the main topic of this document?"
print(f"\nAsking: '{query}'")

query_embedding = get_embedding(query)

# Calculate similarity (Cosine Similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

# Pick top 2 most relevant chunks
top_indices = np.argsort(scores)[::-1][:2]
retrieved_context = "\n\n".join([paragraphs[i] for i in top_indices])

# --- 5. Ask the Model ---
print("--> Step 4: Generating answer from context...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"Answer the user's question using only this context:\n\n{retrieved_context}"
        },
        {"role": "human", "content": query}
    ]
)

print("\n=== Answer ===")
print(response.choices[0].message.content)