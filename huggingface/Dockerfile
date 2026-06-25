FROM python:3.10-slim

WORKDIR /app

# System dependencies for PyMuPDF and FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (needed for BM25)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application files (app.py, rag.py)
COPY . /app

# Ensure cache and upload directories exist
RUN mkdir -p /app/uploads /app/cache /app/model_cache

# Set permissions for Hugging Face (which runs as user 1000)
RUN chmod -R 777 /app/uploads /app/cache /app/model_cache

# Expose port
EXPOSE 7860

# Run with Gunicorn on HF standard port
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
