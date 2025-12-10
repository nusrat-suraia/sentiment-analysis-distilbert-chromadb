"""
FastAPI server for LangChain integration
Required to meet project requirements for LLM framework usage
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb import HttpClient
from chromadb.config import Settings
import uuid

app = FastAPI(title="Sentiment Analysis API")

# Initialize models and connections
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    text: str
    n_results: int = 5

class BatchRequest(BaseModel):
    texts: List[str]

@app.post("/search/semantic")
async def semantic_search(request: QueryRequest):
    """Semantic search using LangChain-like approach"""
    try:
        # Connect to ChromaDB
        client = HttpClient(
            host="chromadb",
            port=8000,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        collection = client.get_or_create_collection("sentiment_data")
        
        # Generate embedding for query
        query_embedding = embedding_model.encode([request.text]).tolist()[0]
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results
        )
        
        return {
            "query": request.text,
            "results": results,
            "total_found": len(results['documents'][0]) if results['documents'] else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/batch")
async def generate_embeddings(request: BatchRequest):
    """Generate embeddings for batch of texts"""
    try:
        embeddings = embedding_model.encode(request.texts).tolist()
        return {
            "texts": request.texts,
            "embeddings": embeddings,
            "dimension": len(embeddings[0]) if embeddings else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)