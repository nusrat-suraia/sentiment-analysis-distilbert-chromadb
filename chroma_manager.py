import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import uuid

class ChromaDBManager:
    def __init__(self, host: str = "localhost", port: int = 8000):
        """Initialize ChromaDB connection"""
        print(f"Connecting to ChromaDB at {host}:{port}")
        
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            print("Connected to ChromaDB successfully!")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            # Try to create persistent client if HTTP fails
            try:
                self.client = chromadb.PersistentClient(
                    path="./chroma_db",
                    settings=Settings(anonymized_telemetry=False)
                )
                print("Connected to local ChromaDB successfully!")
            except Exception as e2:
                print(f"Failed to connect to ChromaDB: {e2}")
                raise
    
    def create_or_get_collection(self, name: str = "sentiment_analysis") -> chromadb.Collection:
        """Create or get a collection"""
        try:
            # Try to create collection
            collection = self.client.create_collection(
                name=name,
                metadata={"description": "Sentiment analysis data"}
            )
            print(f"Created collection: {name}")
        except Exception as e:
            # If collection exists, get it
            print(f"Getting existing collection: {name}")
            collection = self.client.get_collection(name)
        
        return collection
    
    def store_sentiment_results(self, collection: chromadb.Collection, 
                               results: List[Dict[str, Any]],
                               embeddings: List[List[float]]) -> None:
        """Store sentiment analysis results in ChromaDB"""
        if not results or not embeddings:
            print("No data to store")
            return
        
        # Prepare data for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, result in enumerate(results):
            documents.append(result['text'])
            metadatas.append({
                "sentiment": result['sentiment'],
                "confidence": result['confidence'],
                "cleaned_text": result.get('cleaned_text', ''),
                "source": "sentiment_analysis"
            })
            ids.append(f"doc_{i}_{uuid.uuid4().hex[:8]}")
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored {len(documents)} documents in ChromaDB")
    
    def semantic_search(self, collection: chromadb.Collection, 
                       query: str, n_results: int = 5) -> Dict[str, Any]:
        """Perform semantic search on stored documents"""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0,
                        "similarity_score": 1 - (results['distances'][0][i] if results['distances'] else 0)
                    })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return {"query": query, "results": [], "total_found": 0}
    
    def get_collection_stats(self, collection: chromadb.Collection) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = collection.count()
            return {
                "collection_name": collection.name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"collection_name": collection.name, "document_count": 0, "status": "error"}