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

### Running separate modules

Running separate modules is useful for debugging a specific part of the RAG pipeline. For example, the chunking strategy
can be too arbitrary or the embeddings could not capture enough information.

Under `src/core/` you can find the following modules:

- `document.py`: Extracts text from a PDF file and chunks it into smaller pieces
- `embeddings.py`: Implements ChromaDB vector database and visualizes the chunk embeddings after a downprojection onto
  3D space using UMAP
- `llm.py`: Currently just instantiates ChatOllama and defines prompts
- `rag.py`: Implements the RAG pipeline from user input to answer generation