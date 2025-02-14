# Use a lightweight Python image for building
FROM python:3.12-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to take advantage of Docker cache
COPY requirements.txt .

# Upgrade pip, install Python dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK resources
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng

# Copy sample PDF (prevents copying unnecessary files)
COPY data/pdfs/CV.pdf /app/data/pdfs/CV.pdf

# Copy the rest of the application code
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Start Streamlit when the container runs
CMD ["python", "run.py"]