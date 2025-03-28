import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Dict, List

from src.core.document import (
    AdvancedParagraphChunkStrategy,
    DocumentProcessor,
    PDFLoader,
)
from src.core.models import Chunk
from src.core.retrieval.base import RetrievalStrategy, ScoredChunk
from src.core.retrieval.embedding_strategy import EmbeddingsRetrievalStrategy
from src.core.retrieval.keyphrases_strategy import KeyphraseRetrievalStrategy
from src.core.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FusionRetrieval:
    def __init__(
        self,
        strategies: List[RetrievalStrategy],
        alpha: float = 0.5,
        max_cosine_distance: float = 1.6,
        combined_threshold: float = 0.5,
    ):
        """
        :param strategies: List of retrieval strategies (currently EmbeddingsRetrievalStrategy and KeyphraseRetrievalStrategy).
        :param alpha: Parameter between 0 and 1 to control the weight of embeddings vs. keyphrases. 1 means only embeddings.
        :param max_cosine_distance: Chunks above this distance are considered irrelevant.
        """
        self.strategies = strategies
        self.alpha = alpha
        self.max_cosine_distance = max_cosine_distance
        self.combined_threshold = combined_threshold

    def retrieve(self, query: str, top_k: int) -> List[ScoredChunk]:
        all_results = self._run_strategies(query, top_k)

        fused_results = self._fuse_scores(all_results)
        if not fused_results:
            return []

        filtered_results = self._filter_results_by_threshold(
            fused_results, self.max_cosine_distance, self.combined_threshold
        )

        return sorted(filtered_results, key=lambda r: r.combined_score, reverse=True)

    def _run_strategies(self, query: str, top_k: int) -> List[ScoredChunk]:
        """Run all retrieval strategies concurrently and flatten the results."""
        with ThreadPoolExecutor() as executor:
            results_lists = [
                executor.submit(strategy.retrieve, query, top_k).result()
                for strategy in self.strategies
            ]
        return list(chain.from_iterable(results_lists))

    def _fuse_scores(self, results: List[ScoredChunk]) -> List[ScoredChunk]:
        """
        Combine partial scores from different retrieval strategies into a single combined score.

        Steps:
          1. Group results by chunk_id.
          2. For each chunk, build raw score dictionaries:
             - For embedding scores (lower is better), if missing assign a penalty equal to max observed.
             - For keyphrase scores, missing values default to 0.
          3. Normalize each score dictionary.
          4. Fuse them via a weighted sum.
        """
        # 1. Group partial results by chunk_id.
        groups = defaultdict(lambda: {"emb": None, "kp": 0})
        for r in results:
            if r.embedding_score is not None:
                groups[r.chunk_id]["emb"] = r.embedding_score
            if r.keyphrase_score is not None:
                groups[r.chunk_id]["kp"] = r.keyphrase_score

        # 2. Build raw score dictionaries.
        emb_values = [
            data["emb"] for data in groups.values() if data["emb"] is not None
        ]
        default_emb = max(emb_values) if emb_values else 1
        emb_raw = {
            cid: (data["emb"] if data["emb"] is not None else default_emb)
            for cid, data in groups.items()
        }
        kp_raw = {cid: data["kp"] for cid, data in groups.items()}

        logger.info(f"Emb Raw: {emb_raw}")
        logger.info(f"KP Raw: {kp_raw}")

        # 3. Normalize scores.
        norm_emb = self._normalize_scores(emb_raw, lower_is_better=True)
        norm_kp = self._normalize_scores(kp_raw, lower_is_better=False)

        logger.info(f"Norm Emb: {norm_emb}")
        logger.info(f"Norm KP: {norm_kp}")

        # 4. Fuse scores.
        fused_results = []
        for cid in groups:
            combined_score = (
                self.alpha * norm_emb[cid] + (1 - self.alpha) * norm_kp[cid]
            )
            fused_results.append(
                ScoredChunk(
                    chunk_id=cid,
                    embedding_score=emb_raw[cid],
                    keyphrase_score=kp_raw[cid],
                    combined_score=combined_score,
                )
            )

        logger.info(f"Fused Results: {fused_results}")

        return fused_results

    @staticmethod
    def _normalize_scores(
        raw_scores: Dict[str, float], lower_is_better: bool
    ) -> Dict[str, float]:
        """
        Normalize scores using min-max normalization to [0,1].
        If lower_is_better is True, the normalized value is inverted.
        If only one score exists, all normalized scores are set to 1.
        """
        scores = list(raw_scores.values())
        min_val, max_val = min(scores), max(scores)
        if min_val is None or max_val is None:
            raise ValueError("Scores cannot be None.")

        if max_val == min_val:
            return {cid: 1.0 for cid in raw_scores}

        norm_scores = {}
        for cid, score in raw_scores.items():
            if lower_is_better:
                norm_scores[cid] = 1 - ((score - min_val) / (max_val - min_val))
            else:
                norm_scores[cid] = (score - min_val) / (max_val - min_val)

        return norm_scores

    @staticmethod
    def _filter_results_by_threshold(
        fused_results: List[ScoredChunk],
        max_cosine_distance: float,
        combined_threshold: float,
    ) -> List[ScoredChunk]:
        """
        Filter fused results so that a chunk is kept only if:
          1. Its raw keyphrase score is greater than 0.
          2. Its raw embedding score is below max_cosine_distance.
          3. Its fused (combined) score is at least combined_threshold.
        """
        filtered = []
        for r in fused_results:
            if r.keyphrase_score <= 0:
                continue
            if r.embedding_score > max_cosine_distance:
                continue
            if r.combined_score < combined_threshold:
                continue
            filtered.append(r)
        return filtered


# if __name__ == "__main__":
#     start = time.time()
#     pdf_path = Path("../../data/pdfs/microstepexample.pdf")
#     pdf_loader = PDFLoader()
#     advanced_paragraph_chunk_strategy = AdvancedParagraphChunkStrategy()
#     processor_paragraph = DocumentProcessor(
#         loader=pdf_loader, chunk_strategy=advanced_paragraph_chunk_strategy
#     )
#     chunks = processor_paragraph.process_pdf(pdf_path)
#
#     logger.info(f"Number of chunks: {len(chunks)}")
#     for i, chunk in enumerate(chunks):
#         logger.info(f"--- Chunk {i + 1} ---")
#         logger.info(chunk)
#
#     documents = [chunk.text for chunk in chunks]
#     metadatas = [
#         {
#             k: v
#             for k, v in {
#                 "pdf_name": chunk.pdf_name,
#                 "pdf_page": chunk.pdf_page,
#                 "section_name": chunk.section_name,
#                 "subsection_name": chunk.subsection_name,
#                 "chunk_type": chunk.chunk_type,
#             }.items()
#             if v is not None
#         }
#         for chunk in chunks
#     ]
#     ids = [chunk.id for chunk in chunks]
#
#     vector_store = VectorStore(collection_name="local-rag")
#     vector_store.create_vector_db(documents=documents, metadatas=metadatas, ids=ids)
#
#     embedding_strategy = EmbeddingsRetrievalStrategy(vector_store)
#     keyphrase_strategy = KeyphraseRetrievalStrategy(chunks)
#
#     fusion_retriever = FusionRetrieval(
#         strategies=[embedding_strategy, keyphrase_strategy],
#         alpha=0.5,
#         max_cosine_distance=1.6,
#         combined_threshold=0.4,
#     )
#
#     end_time = time.time()
#     print(f"Time to create chunks: {end_time - start}")
#
#     new_start = time.time()
#
#     query = """What is the IMS AWOS User Interface about?"""
#
#     top_k = 7  # Maybe a bit more to make sure we get enough relevant chunks.
#     fused_results = fusion_retriever.retrieve(query, top_k)
#
#     new_end = time.time()
#     print(f"Time to retrieve: {new_end - new_start}")
#
#     chunk_dict = {chunk.id: chunk for chunk in chunks}
#     logger.info("\n\n\nFused Retrieval Results:")
#     for res in fused_results:
#         logger.info(
#             f"Chunk ID: {res.chunk_id} | Combined Score: {res.combined_score:.3f}"
#         )
#         logger.info(chunk_dict.get(res.chunk_id))
#         logger.info("")
#
#     vector_store.delete_collection()


import json
import logging
import time
from pathlib import Path
from typing import List

# Import your modules/classes:
# from your_module import PDFLoader, DocumentProcessor, AdvancedParagraphChunkStrategy
# from your_module import VectorStore, EmbeddingsRetrievalStrategy, KeyphraseRetrievalStrategy, FusionRetrieval, Chunk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_chunks_from_vector_store(vector_store: VectorStore) -> List[Chunk]:
    """
    Loads serialized chunk data from the vector store's metadata and rehydrates chunk objects.
    """
    logger.info("Loading chunks from existing vector DB metadata...")
    result = vector_store.collection.get()
    metadatas = result.get("metadatas", [])
    documents = result.get("documents", [])
    ids = result.get("ids", [])
    loaded_chunks = []
    for i, md in enumerate(metadatas):
        if "chunk_data" in md:
            try:
                chunk_dict = json.loads(md["chunk_data"])
                # Ensure consistency with current document text and id
                chunk_dict.setdefault("text", documents[i])
                chunk_dict.setdefault("id", ids[i])
                chunk = Chunk.from_dict(chunk_dict)
                loaded_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error loading chunk {i}: {e}")
        else:
            logger.warning(f"No chunk_data found in metadata for document {i}")
    logger.info(f"Loaded {len(loaded_chunks)} chunks into memory.")
    return loaded_chunks


def create_vector_store_from_pdf(
    pdf_path: Path, vector_store: VectorStore
) -> List[Chunk]:
    """
    Processes the PDF, creates chunks, and populates the vector store with serialized chunk data.
    Returns the list of chunks.
    """
    logger.info("Processing PDF to generate chunks...")
    pdf_loader = PDFLoader()  # adjust if you need an image transcriber, etc.
    chunk_strategy = AdvancedParagraphChunkStrategy()
    processor = DocumentProcessor(loader=pdf_loader, chunk_strategy=chunk_strategy)
    chunks = processor.process_pdf(pdf_path)
    logger.info(f"Number of chunks generated: {len(chunks)}")

    # Log chunks for debugging
    for i, chunk in enumerate(chunks):
        logger.info(f"--- Chunk {i + 1} ---")
        logger.info(chunk)

    # Prepare documents and metadata (including serialized chunk data)
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "pdf_name": chunk.pdf_name or "",
            "pdf_page": chunk.pdf_page or "",
            "section_name": chunk.section_name or "",
            "subsection_name": chunk.subsection_name or "",
            "chunk_type": chunk.chunk_type or "",
            "chunk_data": json.dumps(chunk.to_dict()),
        }
        for chunk in chunks
    ]

    ids = [chunk.id for chunk in chunks]

    vector_store.create_vector_db(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Vector DB created with chunk data.")
    return chunks


def main():
    start = time.time()
    pdf_path = Path("../../data/pdfs/microstepexample.pdf")
    vector_store = VectorStore(collection_name="local-rag")

    # First, check if the vector store already exists
    if vector_store.collection_exists():
        logger.info("Vector store already exists. Attempting to load chunks from DB.")
        chunks = load_chunks_from_vector_store(vector_store)
        # If no chunks are loaded, fall back to reprocessing the PDF.
        if not chunks:
            logger.warning(
                "No chunk data found in DB; reprocessing PDF to generate chunks."
            )
            chunks = create_vector_store_from_pdf(pdf_path, vector_store)
    else:
        logger.info("Vector store does not exist. Processing PDF to generate chunks.")
        chunks = create_vector_store_from_pdf(pdf_path, vector_store)

    # Build retrieval strategies using the (re)loaded chunks
    embedding_strategy = EmbeddingsRetrievalStrategy(vector_store)
    keyphrase_strategy = KeyphraseRetrievalStrategy(chunks)

    fusion_retriever = FusionRetrieval(
        strategies=[embedding_strategy, keyphrase_strategy],
        alpha=0.5,
        max_cosine_distance=1.6,
        combined_threshold=0.4,
    )

    end = time.time()
    logger.info(f"Setup completed in {end - start:.2f} seconds.")

    # Now run a query (this should be fast if the vector store is populated)
    query = "What is the IMS AWOS User Interface about?"
    top_k = 7  # Retrieve a few relevant chunks
    new_start = time.time()
    fused_results = fusion_retriever.retrieve(query, top_k)
    new_end = time.time()
    logger.info(f"Time to retrieve: {new_end - new_start:.2f} seconds.")

    # Display results (mapping chunk IDs to chunk objects for clarity)
    chunk_dict = {chunk.id: chunk for chunk in chunks}
    logger.info("Fused Retrieval Results:")
    for res in fused_results:
        logger.info(
            f"Chunk ID: {res.chunk_id} | Combined Score: {res.combined_score:.3f}"
        )
        logger.info(chunk_dict.get(res.chunk_id))
        logger.info("")

    # Do not delete the collection if you want persistence:
    # vector_store.delete_collection()


if __name__ == "__main__":
    main()
