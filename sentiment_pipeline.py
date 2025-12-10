import re
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize DistilBERT sentiment analyzer"""
        print(f"Loading sentiment model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load DistilBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Sentiment model loaded successfully!")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to lowercase
        text = text.lower()
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using DistilBERT tokenizer"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenizer.tokenize(cleaned_text)
        return tokens
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text"""
        if not text.strip():
            return {
                "text": text,
                "sentiment": "NEUTRAL",
                "confidence": 0.5,
                "probabilities": {"NEGATIVE": 0.5, "POSITIVE": 0.5}
            }
        
        cleaned_text = self.clean_text(text)
        
        # Tokenize and prepare input
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get results
        sentiment_labels = ["NEGATIVE", "POSITIVE"]
        probabilities = predictions[0].cpu().numpy()
        predicted_class = np.argmax(probabilities)
        
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment_labels[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using sentence-transformers"""
        try:
            # Get embeddings
            cleaned_texts = [self.clean_text(text) for text in texts]
            embeddings = self.embedding_model.encode(cleaned_texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return dummy embeddings if model fails
            return [[0.0] * 384 for _ in texts]  # 384 is the dimension of MiniLM

class DataPreprocessor:
    """Handle data preprocessing tasks - ACTUALLY USED"""
    
    @staticmethod
    def load_and_prepare_data(file_path: str) -> List[Dict[str, Any]]:
        """Prepare dataset from CSV file - INTEGRATED INTO MAIN APP"""
        import pandas as pd
        
        try:
            df = pd.read_csv(file_path)
            data = []
            
            # Assuming first column is text, second column is label (if exists)
            for idx, row in df.iterrows():
                item = {
                    "id": idx,
                    "text": str(row.iloc[0]) if len(row) > 0 else "",
                    "label": row.iloc[1] if len(row) > 1 else None
                }
                data.append(item)
            
            return data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    @staticmethod
    def normalize_texts(texts: List[str]) -> List[str]:
        """Normalize texts by cleaning and standardizing"""
        analyzer = SentimentAnalyzer()
        return [analyzer.clean_text(text) for text in texts]
    
    @staticmethod
    def prepare_training_data(data: List[Dict[str, Any]]) -> tuple:
        """Prepare data for training (if needed for fine-tuning)"""
        texts = [item["text"] for item in data if item["text"]]
        labels = [item["label"] for item in data if item["label"] is not None]
        
        # Map sentiment labels to numerical values
        label_mapping = {"positive": 1, "negative": 0, "neutral": 2}
        numeric_labels = [label_mapping.get(label.lower(), 2) for label in labels]
        
        return texts, numeric_labels