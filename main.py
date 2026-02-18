import os
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from data import documents

# ==========================
# CONFIG
# ==========================
EMBED_MODEL = "text-embedding-ada-002"
RERANK_MODEL = "gpt-4o-mini"
TOTAL_DOCS = len(documents)

# ==========================
# AI PIPE CLIENT
# ==========================
client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)


app = FastAPI()

# ==========================
# Compute & Cache Embeddings
# ==========================
def compute_embeddings(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in response.data]

print("Computing document embeddings...")
doc_embeddings = compute_embeddings([doc["content"] for doc in documents])
print("Embeddings ready.")

# ==========================
# Request Model
# ==========================
class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

# ==========================
# Utilities
# ==========================
def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]

def rerank_with_llm(query, docs):
    scores = []

    for doc in docs:
        prompt = f"""
Query: "{query}"
Document: "{doc['content']}"

Rate relevance from 0-10.
Respond with only a number.
"""
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        score = float(response.choices[0].message.content.strip())
        scores.append(score / 10)  # normalize 0-1

    return scores

# ==========================
# SEARCH ENDPOINT
# ==========================
@app.post("/semantic-search")
def semantic_search(request: SearchRequest):
    start_time = time.time()

    if not request.query.strip():
        return {
            "results": [],
            "reranked": False,
            "metrics": {
                "latency": 0,
                "totalDocs": TOTAL_DOCS
            }
        }

    # Embed query
    query_embedding = compute_embeddings([request.query])[0]

    # Cosine similarity search
    similarities = cosine_similarity(
        [query_embedding],
        doc_embeddings
    )[0]

    # Get top K
    top_indices = np.argsort(similarities)[::-1][:request.k]
    top_docs = [documents[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    normalized_scores = normalize(top_scores)

    results = []
    for doc, score in zip(top_docs, normalized_scores):
        results.append({
            "id": doc["id"],
            "score": round(float(score), 4),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    # ================= RERANK =================
    if request.rerank:
        rerank_scores = rerank_with_llm(request.query, top_docs)

        for i in range(len(results)):
            results[i]["score"] = round(float(rerank_scores[i]), 4)

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results = results[:request.rerankK]

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": results,
        "reranked": request.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": TOTAL_DOCS
        }
    }
