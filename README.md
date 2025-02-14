# Chat with PDF Locally Using Ollama + LangChain

Example adapted from [tonykipkemboi/ollama_pdf_rag](https://github.com/tonykipkemboi/ollama_pdf_rag).

---

## Prerequisites

1. **Install Python 3.12**
    - **Windows**: Download from [python.org](https://www.python.org/downloads/) and check "Add Python to PATH."
    - **macOS**: Install via [Homebrew](https://brew.sh/)
      ```bash
      brew install python@3.12
      ```
    - Or manually download from [python.org](https://www.python.org/downloads/).


2. **Install Ollama**  (an open-source AI model server)
    - See [Ollama’s official site](https://ollama.ai) for instructions.
    - Pull required models (example):
      ```bash
      ollama pull llama3.2  # Or other model for generation. This one is 2GB in size.
      ollama pull nomic-embed-text  # For vector embeddings
      ```


3. **Install Poppler** (for PDF processing)
    - **Windows**:
        1. Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases).
        2. Unzip; add the `bin` folder path to your system’s PATH.
    - **macOS**:
      ```bash
      brew install poppler
      ```


4. **Install Tesseract** (for OCR on scanned PDFs)
    - **Windows**:
        1. Download from [UB Mannheim’s Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).
        2. Install; add Tesseract to your system’s PATH.
    - **macOS**:
      ```bash
      brew install tesseract
      ```


5. **Download NLTK data**
    - Run in a Python shell:
      ```python
      import nltk
      nltk.download('punkt')
      nltk.download('averaged_perceptron_tagger')
      ```

---

## Setup and Installation

1. **Create & activate a virtual environment**
    - **Windows**:
      ```bash
      python -m venv .venv
      .\.venv\Scripts\activate
      ```
    - **macOS/Linux**:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

2. **Install requirements**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

---

## Running the streamlit demo

```bash
python run.py
```

Then open your browser to `http://localhost:8501`

---

# 🐳 Running the Chatbot with Docker

If you prefer **not** to install dependencies manually, you can run the chatbot inside a **Docker container**.

### **1️⃣ Install Docker Desktop**
1. **Download & Install Docker Desktop** from [here](https://www.docker.com/products/docker-desktop).
2. **Create a free Docker account** (if you don’t have one).
3. **Ensure Docker is running** before proceeding.

### **2️⃣ Download the Pre-Built Docker Image**
Instead of building the image yourself, you can **download it from Google Drive**:
- [👉 Download Docker Image Here](TODO: INSERT GOOGLE DRIVE LINK)

Once downloaded, **move it to a convenient location**.

### **3️⃣ Load the Docker Image**
Open a **terminal (PowerShell or Command Prompt)** and run:
```bash
docker load -i path/to/chatbot-demo.tar
```

### **4️⃣ Run the Chatbot**
Run the following command to start the chatbot:
```bash
docker run -p 8501:8501 -e OLLAMA_HOST="http://host.docker.internal:11434" --add-host=host.docker.internal:host-gateway drockdocker/microstep-chatbot-docker-repo:latest
```

If Ollama is not running, start it by simply searching for Ollama and clicking on the app or running:
```bash
ollama serve
```
Then, visit http://localhost:8501 in your browser to start using the chatbot.
In the Streamlit demo, toggle "Use sample PDF (CV.pdf)" to chat with the bot using a sample PDF.
Keep in mind that the app needs to load the pdf file (5sec), chunk it, and generate embeddings for each chunk using your laptop's CPU. This process can take a minute or two, so please be patient.

---

### Running separate modules

Running separate modules is useful for debugging a specific part of the RAG pipeline. For example, the chunking strategy
can be too arbitrary or the embeddings could not capture enough information.

Under `src/core/` you can find the following modules:

- `document.py`: Extracts text from a PDF file and chunks it into smaller pieces
- `embeddings.py`: Implements ChromaDB vector database and visualizes the chunk embeddings after a downprojection onto
  3D space using UMAP
- `llm.py`: Currently just instantiates ChatOllama and defines prompts
- `rag.py`: Implements the RAG pipeline from user input to answer generation

---

# ❓ Troubleshooting
If you face any issues:

1. Ensure Docker Desktop is running before loading the image.
2. Check Ollama status with `curl http://localhost:11434/api/tags`