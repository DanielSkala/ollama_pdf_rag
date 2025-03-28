import base64
import io
import logging
import os
import tempfile
import time
from pathlib import Path

import pdfplumber
import streamlit as st
from document import AdvancedParagraphChunkStrategy, ImageTranscriber, PDFLoader
from dotenv import dotenv_values
from llm import LLMManager
from query_expansion.compression_upscaling import CompressionUpscalingStrategy
from query_expansion.expansion_module import QueryExpansionModule
from rag import RAG

ENV_VARS = dotenv_values(".env.local")
OPENAI_API_KEY = ENV_VARS["OPENAI_API_KEY"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pdf_base64(pdf_bytes: bytes) -> str:
    """Encodes PDF bytes into base64."""
    return base64.b64encode(pdf_bytes).decode("utf-8")


def render_pdf(pdf_bytes: bytes) -> str:
    """Returns HTML code to display a PDF in an iframe."""
    base64_pdf = get_pdf_base64(pdf_bytes)
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800"></iframe>'
    return pdf_display


def load_rag_system(
    model: str,
    pdf_path: str,
    alpha: float,
    max_cosine_distance: float,
    combined_threshold: float,
    min_words: int,
    max_words: int,
) -> RAG:
    """
    Initializes and returns the RAG system based on provided hyperparameters.
    This function is cached so that heavy processing occurs only once.
    """
    # Initialize components

    # Always use gpt-4o for image transcription
    image_transcriber = ImageTranscriber(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

    pdf_loader = PDFLoader(image_transcriber=image_transcriber)
    pdf_path_obj = Path(pdf_path)

    chunk_strategy = AdvancedParagraphChunkStrategy()

    if model in ["gpt-4o", "gpt-4o-mini"]:
        llm_manager = LLMManager(
            model_name=model, provider="openai", openai_api_key=OPENAI_API_KEY
        )
    else:
        llm_manager = LLMManager(model_name=model, provider="ollama")

    compression_strategy = CompressionUpscalingStrategy(
        llm_manager=llm_manager, min_words=min_words, max_words=max_words
    )
    query_expansion_module = QueryExpansionModule(strategy=compression_strategy)

    # Create the RAG instance
    rag_system = RAG(
        pdf_path=pdf_path_obj,
        llm_manager=llm_manager,
        query_expansion_module=query_expansion_module,
        pdf_loader=pdf_loader,
        chunk_strategy=chunk_strategy,
        fusion_retrieval_config={
            "alpha": alpha,
            "max_cosine_distance": max_cosine_distance,
            "combined_threshold": combined_threshold,
        },
        vector_collection_name="local-rag",
    )

    # Initialize the vector store (will load persisted chunks if available)
    rag_system.initialize_vector_store()
    return rag_system


# --- Streamlit App Layout ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.sidebar.title("Configuration")
st.sidebar.image("data/figures/chatbot_architecture.png")

# Sidebar: Hyperparameter settings
model = st.sidebar.selectbox(
    "Select Model", options=["gpt-4o", "gpt-4o-mini", "llama3.2"], index=0
)

alpha = st.sidebar.slider(
    "0 Embeddings, 1 Keywords", min_value=0.0, max_value=1.0, value=0.5, step=0.1
)

max_cosine_distance = st.sidebar.number_input(
    "Max Cosine Distance", min_value=0.0, value=1.6, step=0.1
)
combined_threshold = st.sidebar.number_input(
    "Combined Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.1
)
min_words = st.sidebar.number_input(
    "Min Words (Query Expansion)", min_value=1, value=10, step=1
)
max_words = st.sidebar.number_input(
    "Max Words (Query Expansion)", min_value=1, value=50, step=1
)

# Sidebar: PDF selection
pdf_source = st.sidebar.radio("Select PDF Source", options=["Sample", "Upload"])
sample_pdf_path = "data/pdfs/microstepexample.pdf"

if pdf_source == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
else:
    uploaded_file = None

# Main layout: split into two columns
col1, col2 = st.columns(2)

# Left column: display the PDF document
with col1:
    st.header("PDF Document")
    if pdf_source == "Upload" and uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        # Save a temporary PDF file if needed
        temp_pdf_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        # with open(temp_pdf_path, "wb") as f:
        #     f.write(pdf_bytes)
        active_pdf_path = temp_pdf_path

        # Convert each page to an image with pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            st.session_state["pdf_pages"] = [
                page.to_image().original for page in pdf.pages
            ]

    else:
        # Use sample PDF
        with open(sample_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            st.session_state["pdf_pages"] = [
                page.to_image().original for page in pdf.pages
            ]
        active_pdf_path = sample_pdf_path

    # Now display the pages
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        zoom_level = st.sidebar.slider(
            "Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider",
        )
        with col1:
            with st.container(height=700, border=False):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

# Right column: chat interface for the RAG chatbot
with col2:
    st.header("RAG Chatbot")
    user_query = st.text_area(
        "Enter your query:",
        height=100,
        value="What is the frequency of data updates, and how often does the Cloud Base Display refresh the information from the ceilometer?",
    )
    if st.button("Submit Query"):
        start_time = time.time()
        if user_query.strip():
            with st.spinner("Generating answer..."):
                try:
                    rag_system = load_rag_system(
                        model,
                        active_pdf_path,
                        alpha,
                        max_cosine_distance,
                        combined_threshold,
                        min_words,
                        max_words,
                    )
                    answer = rag_system.generate_answer(query=user_query)
                except Exception as e:
                    st.warning("Please, try again, this is a known bug...")
                    answer = None
            end_time = time.time()

            st.subheader("Answer")
            st.write(answer)

            st.markdown(
                f'<span style="color: #4CAF50;">{end_time - start_time:.2f} seconds</span>',
                unsafe_allow_html=True,
            )

        else:
            st.warning("Please enter a query.")

# Sidebar: Additional actions
st.sidebar.header("Actions")
if st.sidebar.button("Delete Vector Store"):
    rag_system = load_rag_system(
        model,
        active_pdf_path,
        alpha,
        max_cosine_distance,
        combined_threshold,
        min_words,
        max_words,
    )
    rag_system.cleanup()
    st.success("Vector store deleted.")
