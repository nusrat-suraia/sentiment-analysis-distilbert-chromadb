FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data

# Expose port
EXPOSE 7860
EXPOSE 8001


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHROMA_HOST=chromadb
ENV CHROMA_PORT=8000

# Run the application
CMD ["python", "app.py"]    


