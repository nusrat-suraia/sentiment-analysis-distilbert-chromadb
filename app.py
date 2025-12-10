import os
import gradio as gr
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
from datetime import datetime
import requests

from sentiment_pipeline import SentimentAnalyzer, DataPreprocessor
from chroma_manager import ChromaDBManager

# Initialize components
print("=" * 50)
print("Initializing Sentiment Analysis Application...")
print("=" * 50)

sentiment_analyzer = SentimentAnalyzer()

# Get ChromaDB host from environment variables (Docker Compose sets this)
chroma_host = os.getenv("CHROMA_HOST", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
api_url = os.getenv("API_URL", "http://localhost:8001")

# Try to connect to ChromaDB, fall back to local if needed
try:
    chroma_manager = ChromaDBManager(host=chroma_host, port=chroma_port)
    collection = chroma_manager.create_or_get_collection("sentiment_data")
    print(f"Connected to ChromaDB at {chroma_host}:{chroma_port} successfully!")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB: {e}")
    print("Running in local mode without ChromaDB...")
    chroma_manager = None
    collection = None

# Check API service
try:
    response = requests.get(f"{api_url}/health")
    api_available = response.status_code == 200
    print(f"API service available: {api_available}")
except:
    api_available = False
    print("API service not available")

# Sample data for demonstration
SAMPLE_TEXTS = [
    "I absolutely love this product! It's amazing and works perfectly.",
    "The service was terrible and the staff was rude.",
    "It's okay, nothing special but gets the job done.",
    "Worst experience of my life. Never buying again.",
    "Excellent quality and fast delivery. Highly recommended!"
]

# Initialize with sample data if ChromaDB is available
if chroma_manager and collection:
    print("Initializing with sample data...")
    try:
        sample_results = sentiment_analyzer.batch_analyze(SAMPLE_TEXTS)
        sample_embeddings = sentiment_analyzer.get_text_embeddings(SAMPLE_TEXTS)
        chroma_manager.store_sentiment_results(collection, sample_results, sample_embeddings)
        print("Sample data initialized!")
    except Exception as e:
        print(f"Warning: Could not initialize sample data: {e}")

# Gradio UI Functions
def analyze_single_text(text: str) -> Dict[str, Any]:
    """Analyze sentiment of a single text"""
    if not text.strip():
        return {"error": "Please enter some text to analyze"}
    
    # Analyze sentiment
    result = sentiment_analyzer.analyze_sentiment(text)
    
    # Store in ChromaDB if available
    if chroma_manager and collection:
        try:
            embeddings = sentiment_analyzer.get_text_embeddings([text])
            chroma_manager.store_sentiment_results(collection, [result], embeddings)
        except Exception as e:
            print(f"Warning: Could not store in ChromaDB: {e}")
    
    return result

def analyze_batch_texts(texts: str) -> Dict[str, Any]:
    """Analyze sentiment of multiple texts"""
    if not texts.strip():
        return {"error": "Please enter texts to analyze"}
    
    # Split by newlines
    text_list = [t.strip() for t in texts.split('\n') if t.strip()]
    
    if not text_list:
        return {"error": "No valid texts found"}
    
    # Analyze in batches
    results = sentiment_analyzer.batch_analyze(text_list[:20])  # Limit to 20
    
    # Store in ChromaDB if available
    if chroma_manager and collection:
        try:
            embeddings = sentiment_analyzer.get_text_embeddings(text_list[:20])
            chroma_manager.store_sentiment_results(collection, results, embeddings)
        except Exception as e:
            print(f"Warning: Could not store in ChromaDB: {e}")
    
    # Calculate summary statistics
    sentiments = [r['sentiment'] for r in results]
    positive_count = sentiments.count("POSITIVE")
    negative_count = sentiments.count("NEGATIVE")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    return {
        "summary": {
            "total_texts": len(results),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": (positive_count / len(results)) * 100,
            "negative_percentage": (negative_count / len(results)) * 100,
            "average_confidence": avg_confidence
        },
        "sample_results": results[:3]  # Show first 3 results
    }

def analyze_file(file) -> Dict[str, Any]:
    """Analyze sentiment from uploaded file"""
    if file is None:
        return {"error": "Please upload a file"}
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            # Use first column as text
            texts = df.iloc[:, 0].dropna().astype(str).tolist()
            
            # Use DataPreprocessor class as required
            processed_data = DataPreprocessor.load_and_prepare_data(file.name)
            print(f"Processed {len(processed_data)} records using DataPreprocessor")
            
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            return {"error": "Unsupported file format. Please upload CSV or TXT."}
        
        if not texts:
            return {"error": "No valid text found in the file"}
        
        # Analyze
        results = sentiment_analyzer.batch_analyze(texts[:50])  # Limit to 50
        
        # Store in ChromaDB if available
        if chroma_manager and collection:
            try:
                embeddings = sentiment_analyzer.get_text_embeddings(texts[:50])
                chroma_manager.store_sentiment_results(collection, results, embeddings)
            except Exception as e:
                print(f"Warning: Could not store in ChromaDB: {e}")
        
        # Create summary
        sentiments = [r['sentiment'] for r in results]
        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")
        
        return {
            "file_name": file.name,
            "total_analyzed": len(results),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sample_results": results[:3]  # Show first 3 results
        }
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

def search_similar_texts(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Search for similar texts in the database"""
    if not query.strip():
        return {"error": "Please enter a search query"}
    
    if not chroma_manager or not collection:
        return {"error": "ChromaDB not available. Please run with Docker Compose for full features."}
    
    results = chroma_manager.semantic_search(collection, query, n_results)
    return results

def use_langchain_api(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Use API service for LangChain-like functionality"""
    if not api_available:
        return {"error": "API service not available. Run Docker Compose for full features."}
    
    try:
        response = requests.post(f"{api_url}/search/semantic", 
                                json={"text": query, "n_results": n_results})
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def get_dashboard_stats() -> Dict[str, Any]:
    """Get dashboard statistics"""
    if chroma_manager and collection:
        stats = chroma_manager.get_collection_stats(collection)
    else:
        stats = {"collection_name": "Not Available", "document_count": 0, "status": "ChromaDB not connected"}
    
    return {
        "collection_stats": stats,
        "application_status": "Running" if chroma_manager else "Running (ChromaDB not connected)",
        "api_status": "Available" if api_available else "Not Available",
        "model_loaded": "DistilBERT",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Create visualization functions
def create_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create sentiment distribution chart"""
    labels = ['Positive', 'Negative']
    values = [sentiment_data.get('positive', 0), sentiment_data.get('negative', 0)]
    
    colors = ['#4CAF50', '#F44336']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.3,
        marker=dict(colors=colors)
    )])
    fig.update_layout(
        title="Sentiment Distribution",
        showlegend=True,
        height=300
    )
    return fig

# Gradio Interface
with gr.Blocks(title="Sentiment Analysis Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Sentiment Analysis with DistilBERT")
    gr.Markdown("Analyze text sentiment using DistilBERT transformer model and store results in ChromaDB vector database.")
    
    with gr.Tab("📝 Single Text Analysis"):
        with gr.Row():
            with gr.Column(scale=2):
                single_text = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type your text here... Example: 'I love this amazing product!'",
                    lines=5,
                    info="Enter text to analyze its sentiment"
                )
                single_analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
                
                gr.Examples(
                    examples=[
                        ["I love this product! It's amazing and works perfectly."],
                        ["The service was terrible and the staff was rude."],
                        ["Good value for money, but could be better."]
                    ],
                    inputs=single_text,
                    label="Try these examples:"
                )
            
            with gr.Column(scale=3):
                sentiment_output = gr.JSON(label="Sentiment Analysis Result")
        
        def update_single_analysis(text):
            result = analyze_single_text(text)
            return result
        
        single_analyze_btn.click(
            update_single_analysis,
            inputs=single_text,
            outputs=sentiment_output
        )
    
    with gr.Tab("📄 Batch Analysis"):
        with gr.Row():
            with gr.Column():
                batch_texts = gr.Textbox(
                    label="Enter Multiple Texts (one per line)",
                    placeholder="Enter each text on a new line...\n\nExample:\nI love this product!\nTerrible service\nGood value for money",
                    lines=10
                )
                batch_analyze_btn = gr.Button("Analyze Batch", variant="primary")
            
            with gr.Column():
                batch_output = gr.JSON(label="Batch Analysis Results")
        
        def update_batch_analysis(texts):
            result = analyze_batch_texts(texts)
            return result
        
        batch_analyze_btn.click(
            update_batch_analysis,
            inputs=batch_texts,
            outputs=batch_output
        )
    
    with gr.Tab("📁 File Upload"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload CSV or TXT File",
                    file_types=[".csv", ".txt"],
                    # type="filepath"
                )
                gr.Markdown("""
                **File Format:**
                - CSV: First column should contain text
                - TXT: Each line is treated as a separate text
                
                **Preprocessing:** Uses DataPreprocessor class for proper data preparation
                """)
                file_analyze_btn = gr.Button("Analyze File", variant="primary")
            
            with gr.Column():
                file_output = gr.JSON(label="File Analysis Results")
        
        file_analyze_btn.click(
            analyze_file,
            inputs=file_input,
            outputs=file_output
        )
    
    with gr.Tab("🔍 Semantic Search"):
        with gr.Tab("ChromaDB Search"):
            with gr.Row():
                with gr.Column():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Search for similar texts...",
                        lines=2
                    )
                    n_results = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Number of Results"
                    )
                    search_btn = gr.Button("Search in ChromaDB", variant="primary")
                
                with gr.Column():
                    search_output = gr.JSON(label="Search Results")
            
            search_btn.click(
                search_similar_texts,
                inputs=[search_query, n_results],
                outputs=search_output
            )
            
    
    with gr.Tab("📊 Dashboard"):
        with gr.Row():
            with gr.Column():
                stats_btn = gr.Button("Refresh Dashboard", variant="secondary")
                dashboard_stats = gr.JSON(label="System Statistics")
        
        def update_dashboard():
            stats = get_dashboard_stats()
            return stats
        
        stats_btn.click(
            update_dashboard,
            outputs=dashboard_stats
        )
    
    with gr.Tab("ℹ️ About & Help"):
        gr.Markdown("""
        ## Sentiment Analysis Application
        
        **Features:**
        - ✅ Uses DistilBERT transformer model for sentiment analysis
        - ✅ Text preprocessing (cleaning, tokenization, normalization) using DataPreprocessor
        - ✅ ChromaDB vector database integration
        - ✅ LangChain API service for LLM framework integration
        - ✅ Semantic search capabilities
        - ✅ Batch processing and file upload
        - ✅ Docker containerization with three services
        
        **How to Run:**
        
        **With Docker Compose (Required for Full Features):**
        ```bash
        docker-compose up --build
        ```
        Then access at: http://localhost:7860
        
        **Services:**
        1. ChromaDB Vector Database (port 8000)
        2. FastAPI/LangChain Service (port 8001)
        3. Gradio UI Application (port 7860)
        
        **Preprocessing Steps:**
        1. Text cleaning (remove URLs, special characters)
        2. Tokenization using DistilBERT tokenizer
        3. Text normalization (lowercasing, whitespace cleanup)
        4. Embedding generation for vector storage
        5. Dataset preparation using DataPreprocessor
        """)

# Run the application
if __name__ == "__main__":
    print("=" * 50)
    print("Starting Sentiment Analysis Application...")
    print("Access the application at: http://localhost:7860")
    print("=" * 50)
    
    # demo.launch()
    demo.launch(
    server_name="0.0.0.0",  # Required for Docker
    server_port=7860,
    share=False  # Explicitly set to False for Docker
    )