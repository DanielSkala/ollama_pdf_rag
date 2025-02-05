"""Vector embeddings and database functionality."""

import logging
import textwrap
from pathlib import Path
from typing import List

import plotly.graph_objects as go
import umap.umap_ as umap
from document import DocumentProcessor
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings and database operations."""

    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_db = None

    def create_vector_db(
        self, documents: List, collection_name: str = "local-rag"
    ) -> Chroma:
        """Create vector database from documents."""
        try:
            logger.info("Creating vector database")
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
            )
            return self.vector_db
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete vector database collection."""
        if self.vector_db:
            try:
                logger.info("Deleting vector database collection")
                self.vector_db.delete_collection()
                self.vector_db = None
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
                raise


def plot_embeddings(
    chunk_embeddings: List, texts: List, query_embedding: List, user_query: str
) -> None:
    """Plot text embeddings in 3D space with user query."""
    reducer = umap.UMAP(n_components=3)
    chunk_embeddings_3d = reducer.fit_transform(chunk_embeddings)

    wrapped_texts = ["<br>".join(textwrap.wrap(txt, width=80)) for txt in texts]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=chunk_embeddings_3d[:, 0],
            y=chunk_embeddings_3d[:, 1],
            z=chunk_embeddings_3d[:, 2],
            mode="markers",
            name="Chunks",
            customdata=wrapped_texts,
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(size=5, color="blue", opacity=0.8),
        )
    )

    query_embedding_3d = reducer.transform([query_embedding])

    fig.add_trace(
        go.Scatter3d(
            x=[query_embedding_3d[0, 0]],
            y=[query_embedding_3d[0, 1]],
            z=[query_embedding_3d[0, 2]],
            mode="markers",
            name="User Query",
            customdata=[f"User Query: {user_query}"],
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(size=8, color="red"),
        )
    )

    fig.update_layout(
        title="Text Embeddings Visualized in 3D Space (User Query in Red)",
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800,
    )

    fig.show()


if __name__ == "__main__":
    processor = DocumentProcessor()
    pdf_path = Path("../../data/pdfs/cv.pdf")
    documents = processor.load_pdf(pdf_path)
    chunks = processor.split_documents(documents)

    vector_store = VectorStore()
    vector_db = vector_store.create_vector_db(
        documents=chunks, collection_name="local-rag"
    )

    user_query = "What is this AI Quiz and AI Polls about?"
    query_embedding = vector_db._embedding_function.embed_query(user_query)

    embeddings_obj = vector_db._collection.get(include=["embeddings"])
    chunk_embeddings = embeddings_obj["embeddings"]

    texts = [chunk.page_content for chunk in chunks]

    reducer = umap.UMAP(n_components=3)
    chunk_embeddings_3d = reducer.fit_transform(chunk_embeddings)

    wrapped_texts = ["<br>".join(textwrap.wrap(txt, width=80)) for txt in texts]

    plot_embeddings(chunk_embeddings, texts, query_embedding, user_query)
