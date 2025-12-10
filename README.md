# Sentiment Analysis Application with DistilBERT & ChromaDB

## 📋 Project Overview
A complete sentiment analysis system using the **DistilBERT transformer model** (for sentiment classification), **ChromaDB vector database** (for semantic search), and **Docker containerization** (for easy cross-platform deployment). Features a user-friendly Gradio UI and FastAPI backend.


## 🚀 Quick Start Guide
### Prerequisites
- Docker & Docker Compose installed (see [Docker Installation Guide](#4-docker-installation-guide))
- At least 4GB RAM available
- Internet connection (for model/image downloads on first run)


### Step-by-Step Setup
#### Option 1: Using Docker Compose (Recommended—Full Features)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/nusrat-suraia/sentiment-analysis-distilbert-chromadb.git
   cd sentiment-analysis-distilbert-chromadb
   ```

2. **Build and start the system**:
   ```bash
   docker compose up --build
   ```
   - First run takes 5–10 minutes (downloads models/ChromaDB image).
   - Wait for the terminal to show: `Running on http://0.0.0.0:7860` (Gradio UI).

3. **Access the app**:
   - UI: Open `http://localhost:7860` in your browser
   - ChromaDB: Open `http://localhost:8000/docs`
   - API Docs: Open `http://localhost:8001/docs` (test endpoints interactively)


#### Option 2: Run Locally (Limited Features—No ChromaDB)
1. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run ChromaDB locally (optional, for search features)**:
   ```bash
   - chroma run --host 0.0.0.0 --port 8000
   ```
4. **Run the app**:
   ```bash
   python app.py
   ```
   


## ✨ Key Features
1. **Sentiment Analysis**:
   - Classify text as **Positive/Negative** with confidence scores (0–1).
   - Supports:
     - Single text input
     - Batch text input (paste multiple lines)
     - File upload (CSV/TXT—first column = text)

2. **Semantic Search**:
   - Find contextually similar texts (e.g., search "worst purchase" to find negative reviews).
   - Uses **Sentence-BERT embeddings** stored in ChromaDB

3. **User-Friendly UI**:
   - Tabbed Gradio interface (Single/Batch/File Upload/Search).
   - Pre-filled examples (no typing required to test).


## 🛠️ Project File Structure
```
sentiment-analysis-distilbert-chromadb/
├── app.py                  # Gradio frontend UI
├── api_server.py           # FastAPI backend (API endpoints)
├── chroma_manager.py       # ChromaDB handler (embeddings + search)
├── sentiment_pipeline.py   # DistilBERT model + text preprocessing
├── Dockerfile              # Builds app container image
├── docker-compose.yml      # Orchestrates 3 services (ChromaDB/API/UI)
├── requirements.txt        # Python dependencies (models, libraries)
├── data/
│   └── sample_reviews.csv  # Test dataset (e-commerce reviews)
└── README.md               # This guide
```


## 🐳 Docker Installation Guide
### Windows/Mac
1. Download Docker Desktop: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Install and open Docker (wait for the whale icon to stop spinning).
3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```


### Linux (Ubuntu/CentOS)
1. **Install Docker**:
   ```bash
   # Ubuntu
   sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

   # CentOS
   sudo yum update && sudo yum install docker-ce docker-ce-cli containerd.io

   # Start Docker
   sudo systemctl start docker
   sudo systemctl enable docker  # Auto-start on boot
   ```

2. **Install Docker Compose**:
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose

   # Verify
   docker-compose --version
   ```


## 🧩 Troubleshooting
### Issue 1: Docker containers won’t start
- Fix: Ensure Docker Desktop is running (Windows/Mac) or run `sudo systemctl start docker` (Linux).


### Issue 2: Gradio says "ChromaDB not available"
- Fix: Confirm all 3 services are running:
  ```bash
  docker ps  # Look for sentiment-chromadb, sentiment-api, sentiment-ml-app
  ```


### Issue 3: Slow model downloads
- Fix: Add a PyPI mirror to `Dockerfile` (replace the `pip install` line):
  ```dockerfile
  RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
  ```


## 📊 Test Results
- **Sentiment Accuracy**: 90% (18/20 correct on `sample_reviews.csv`).
- **Semantic Search Example**:
  - Query: `"worst purchase"`
  - Top Result: `"Worst purchase of my life. Never buying again."` (similarity score: 0.89)


## 📝 Notes
- This project is built for educational purposes (intermediate NLP/ML deployment).
- For production use: Add authentication, rate limiting, and model fine-tuning.