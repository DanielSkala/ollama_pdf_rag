# Chat with PDF locally using Ollama + LangChain

(Example taken from [this repo](https://github.com/tonykipkemboi/ollama_pdf_rag))

## Getting Started

### Prerequisites

1. **Install Ollama**
   - Visit [Ollama's website](https://ollama.ai) to download and install
   - Pull required models:
     ```bash
     ollama pull llama3.2  # or your preferred model
     ollama pull nomic-embed-text
     ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the streamlit demo
```bash
python run.py
```
Then open your browser to `http://localhost:8501`

### Running separate modules
Running separate modules is useful for debugging a specific part of the RAG pipeline. For example, the chunking strategy can be too arbitrary or the embeddings could not capture enough information.

Under `src/core/` you can find the following modules:
- `document.py`: Extracts text from a PDF file and chunks it into smaller pieces
- `embeddings.py`: Implements ChromaDB vector database and visualizes the chunk embeddings after a downprojection onto 3D space using UMAP
- `llm.py`: Currently just instantiates ChatOllama and defines prompts
- `rag.py`: Implements the RAG pipeline from user input to answer generation

## ⚠️ Troubleshooting

TBD