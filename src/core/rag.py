import json
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
from src.core.models import Chunk
from src.core.prompts import get_system_prompt, get_user_prompt
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
        self.chunk_strategy = (
            chunk_strategy
            if chunk_strategy is not None
            else AdvancedParagraphChunkStrategy()
        )
        self.processor = DocumentProcessor(
            loader=self.pdf_loader, chunk_strategy=self.chunk_strategy
        )
        self.chunks: List[Any] = []
        self.vector_store = VectorStore(collection_name=vector_collection_name)
        self.fusion_config = (
            fusion_retrieval_config if fusion_retrieval_config is not None else {}
        )
        self.fusion_retriever = None

    def _initialize_system(self):
        """
        Processes the document, creates chunks, and builds the vector store and fusion retriever.
        This is used when the vector DB does not yet exist or no chunk metadata is available.
        """
        start_time = time.time()
        logger.info("Processing PDF into chunks...")
        self.chunks = self.processor.process_pdf(self.pdf_path)
        logger.info(f"Document processed into {len(self.chunks)} chunks.")

        # Build vector store using the chunks' text, metadata, and ids.
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
                    "chunk_data": json.dumps(chunk.to_dict()),
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

        # Build the fusion retriever using both the embedding and keyphrase strategies.
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

    def _load_chunks_from_db(self) -> List[Any]:
        """
        Loads serialized chunk data from the vector store's metadata and rehydrates chunk objects.
        """
        logger.info("Loading chunks from existing vector DB metadata...")
        result = self.vector_store.collection.get()
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])
        ids = result.get("ids", [])
        loaded_chunks = []
        for i, md in enumerate(metadatas):
            if "chunk_data" in md:
                try:
                    chunk_dict = json.loads(md["chunk_data"])
                    # Ensure text and id are consistent
                    chunk_dict.setdefault("text", documents[i])
                    chunk_dict.setdefault("id", ids[i])
                    chunk = Chunk.from_dict(chunk_dict)
                    loaded_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error loading chunk {i}: {e}")
            else:
                logger.warning(f"No chunk_data found in metadata for document {i}")
        logger.info(f"Loaded {len(loaded_chunks)} chunks from DB.")
        return loaded_chunks

    def initialize_vector_store(self):
        """
        Initializes the vector store. If the collection already exists, loads the chunks from
        the DB metadata; otherwise, processes the PDF and creates the vector DB.
        Finally, builds the fusion retriever.
        """
        if self.vector_store.collection_exists():
            logger.info("Vector store already exists. Loading chunks from DB metadata.")
            self.chunks = self._load_chunks_from_db()
            if not self.chunks:
                logger.warning("No chunk data loaded from DB. Reprocessing PDF.")
                self._initialize_system()
            else:
                # Build fusion retriever based on loaded chunks.
                embedding_strategy = EmbeddingsRetrievalStrategy(self.vector_store)
                keyphrase_strategy = KeyphraseRetrievalStrategy(self.chunks)
                self.fusion_retriever = FusionRetrieval(
                    strategies=[embedding_strategy, keyphrase_strategy],
                    alpha=self.fusion_config.get("alpha", 0.5),
                    max_cosine_distance=self.fusion_config.get(
                        "max_cosine_distance", 1.6
                    ),
                    combined_threshold=self.fusion_config.get(
                        "combined_threshold", 0.4
                    ),
                )
        else:
            logger.info(
                "Vector store does not exist. Processing PDF and creating vector DB."
            )
            self._initialize_system()

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

    def generate_answer(self, query: str, top_k: int = 7) -> str:
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
                retrieved_texts += str(chunk) + "\n"

        system_prompt = get_system_prompt()
        final_user_prompt = get_user_prompt(query, retrieved_texts)

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
    MODEL = "gpt-4o-mini"  # Or "llama3.2" if using Ollama

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

    rag_system.initialize_vector_store()

    # Process a user query
    user_query = "What is the frequency of data updates, and how often does the Cloud Base Display refresh the information from the ceilometer?"
    # user_query = "Where is Mona Lisa?"

    answer = rag_system.generate_answer(query=user_query)
    print("Generated Answer:\n", answer)

    # To persist the vector store for future runs, do NOT call cleanup.
    # rag_system.cleanup()
