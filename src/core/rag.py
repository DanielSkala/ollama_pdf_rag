import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import dotenv_values

from src.core.document import (
    AdvancedParagraphChunkStrategy,
    DocumentProcessor,
    ImageTranscriber,
    PDFLoader,
)
from src.core.llm import LLMManager
from src.core.query_expansion.compression_upscaling import CompressionUpscalingStrategy
from src.core.query_expansion.expansion_module import QueryExpansionModule
from src.core.retrieval.embedding_strategy import EmbeddingsRetrievalStrategy
from src.core.retrieval.fusion_retrieval import FusionRetrieval
from src.core.retrieval.keyphrases_strategy import KeyphraseRetrievalStrategy
from src.core.retrieval.vector_store import VectorStore

ENV_VARS = dotenv_values("../../.env.local")
OPENAI_API_KEY = ENV_VARS["OPENAI_API_KEY"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAG:
    """
    Retrieval Augmented Generation (RAG) system that ties together document processing,
    query expansion, fusion retrieval, and language model response generation.
    """

    def __init__(
        self,
        pdf_path: Path,
        llm_manager: Any,
        query_expansion_module: QueryExpansionModule,
        pdf_loader: Optional[Any] = None,
        chunk_strategy: Optional[Any] = None,
        fusion_retrieval_config: Optional[Dict[str, Any]] = None,
        vector_collection_name: str = "local-rag",
    ):
        """
        Initializes the RAG system.

        Args:
            pdf_path (Path): Path to the PDF document.
            llm_manager (LLMManager): Instance managing LLM calls.
            query_expansion_module (QueryExpansionModule): Query expansion module.
            pdf_loader (Optional[PDFLoader]): PDF loader; if not provided, a default instance is used.
            chunk_strategy (Optional): Strategy for document chunking; if not provided, a default AdvancedParagraphChunkStrategy is used.
            fusion_retrieval_config (Optional[Dict]): Configuration for FusionRetrieval parameters.
            vector_collection_name (str): Name of the vector store collection.
        """
        self.pdf_path = pdf_path
        self.llm_manager = llm_manager
        self.query_expansion_module = query_expansion_module
        self.pdf_loader = pdf_loader if pdf_loader is not None else PDFLoader()
        self.chunk_strategy = chunk_strategy
        self.processor = DocumentProcessor(
            loader=self.pdf_loader, chunk_strategy=self.chunk_strategy
        )
        self.chunks: List[Any] = []
        self.vector_store = VectorStore(collection_name=vector_collection_name)
        self.fusion_config = fusion_retrieval_config
        self.fusion_retriever = None
        self._initialize_system()

    def _initialize_system(self):
        """Processes the document, creates chunks, initializes vector store and the fusion retriever."""
        start_time = time.time()
        logger.info("Starting document processing...")
        self.chunks = self.processor.process_pdf(self.pdf_path)
        logger.info(f"Document processed into {len(self.chunks)} chunks.")

        # Build vector store using the chunks' text, metadata, and ids
        documents = [chunk.text for chunk in self.chunks]
        metadatas = [
            {
                k: v
                for k, v in {
                    "pdf_name": chunk.pdf_name,
                    "pdf_page": chunk.pdf_page,
                    "section_name": chunk.section_name,
                    "subsection_name": chunk.subsection_name,
                    "chunk_type": chunk.chunk_type,
                }.items()
                if v is not None
            }
            for chunk in self.chunks
        ]
        ids = [chunk.id for chunk in self.chunks]
        self.vector_store.create_vector_db(
            documents=documents, metadatas=metadatas, ids=ids
        )
        logger.info("Vector store created successfully.")

        # Initialize fusion retriever with both embedding and keyphrase strategies
        embedding_strategy = EmbeddingsRetrievalStrategy(self.vector_store)
        keyphrase_strategy = KeyphraseRetrievalStrategy(self.chunks)

        self.fusion_retriever = FusionRetrieval(
            strategies=[embedding_strategy, keyphrase_strategy],
            alpha=self.fusion_config.get("alpha", 0.5),
            max_cosine_distance=self.fusion_config.get("max_cosine_distance", 1.6),
            combined_threshold=self.fusion_config.get("combined_threshold", 0.4),
        )

        end_time = time.time()
        logger.info(f"RAG system initialized in {end_time - start_time:.2f} seconds.")

    def _retrieve(self, query: str, top_k: int = 7) -> List[Any]:
        """
        Expands the query and retrieves relevant document chunks.

        Args:
            query (str): The user's query.
            top_k (int): Number of top chunks to retrieve.

        Returns:
            List: A list of retrieval results.
        """
        logger.info("Expanding query...")
        expanded_query = self.query_expansion_module.expand_query(query)

        logger.info(f"Expanded query: {expanded_query}")
        logger.info("Performing retrieval...")
        results = self.fusion_retriever.retrieve(expanded_query, top_k)

        logger.info(f"Retrieved {len(results)} chunks.")
        return results

    def generate_answer(
        self,
        query: str,
        top_k: int = 7,
    ) -> str:
        """
        Generates an answer by retrieving relevant chunks and calling the LLM.

        Args:
            query (str): The original user query.
            top_k (int): Number of chunks to retrieve for context.

        Returns:
            str: The response generated by the LLM.
        """
        results = self._retrieve(query, top_k)

        retrieved_texts = ""
        for res in results:
            chunk = self._get_chunk_by_id(res.chunk_id)
            if chunk:
                retrieved_texts += chunk.__str__() + "\n"

        system_prompt = "You are a helpful assistant specialized in MicroStep-MIS documentation. Your task is to provide accurate and relevant information based on the user's query and the provided context. The context is a number of chunks that have sections, page numbers, and actual text from the documentation. Based on these chunks, answer the user question as accurately as possible and ALWAYS end your answer by pointing the user to a specific section and pdf page number to read more!"
        final_user_prompt = f"User Query: {query}\n\nRelevant Information (chunks):\n\n{retrieved_texts}User Query: {query}\n\nAnd now provide an answer and also point the user to a specific section and page number to read more about the topic."

        print(final_user_prompt)

        logger.info("Generating response from LLM...")
        response = self.llm_manager.generate_response(
            system_prompt=system_prompt, user_prompt=final_user_prompt
        )

        logger.info("LLM response generated.")
        return response

    def _get_chunk_by_id(self, chunk_id: str) -> Optional[Any]:
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        logger.warning(f"Chunk with ID {chunk_id} not found.")
        return None

    def cleanup(self):
        self.vector_store.delete_collection()
        logger.info("Vector store cleaned up.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Setup configuration and credentials
    MODEL = "gpt-4o"  # Or "llama3.2" if using Ollama

    # Initialize the image transcriber (used by the PDFLoader)
    image_transcriber = ImageTranscriber(openai_api_key=OPENAI_API_KEY, model=MODEL)
    pdf_loader = PDFLoader(image_transcriber=image_transcriber)
    pdf_path = Path("../../data/pdfs/microstepexample.pdf")

    # Choose the chunking strategy
    advanced_paragraph_chunk_strategy = AdvancedParagraphChunkStrategy()

    # Initialize the LLM manager
    llm_manager = LLMManager(
        model_name=MODEL, provider="openai", openai_api_key=OPENAI_API_KEY
    )

    # Set up the query expansion module with a compression strategy
    compression_strategy = CompressionUpscalingStrategy(
        llm_manager=llm_manager, min_words=10, max_words=50
    )
    expansion_module = QueryExpansionModule(strategy=compression_strategy)

    # Create the RAG system instance
    rag_system = RAG(
        pdf_path=pdf_path,
        llm_manager=llm_manager,
        query_expansion_module=expansion_module,
        pdf_loader=pdf_loader,
        chunk_strategy=advanced_paragraph_chunk_strategy,
        fusion_retrieval_config={
            "alpha": 0.5,
            "max_cosine_distance": 1.6,
            "combined_threshold": 0.4,
        },
        vector_collection_name="local-rag",
    )

    # Process a user query
    user_query = "What is the frequency of data updates, and how often does the Cloud Base Display refresh the information from the ceilometer?"
    answer = rag_system.generate_answer(query=user_query)
    print("Generated Answer:\n", answer)

    # Clean up resources
    rag_system.cleanup()
