from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    BULLETPOINTS = "bulletpoints"
    IMAGE_CAPTION = "image_caption"


class Chunk(BaseModel):
    id: str
    pdf_name: str
    pdf_page: int
    section_name: Optional[str] = None
    subsection_name: Optional[str] = None
    chunk_type: ChunkType
    text: str
    keywords: Optional[list] = None

    def __repr__(self) -> str:
        return (
            f"CHUNK ID '{self.id}':\n"
            f"Page={self.pdf_page},\n"
            f"Section={self.section_name!r},\n"
            f"Text={self.text!r}\n"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self):
        """Return a dictionary of only serializable fields."""
        return {
            "id": self.id,
            "pdf_name": self.pdf_name,
            "pdf_page": self.pdf_page,
            "section_name": self.section_name,
            "subsection_name": self.subsection_name,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d.get("id"),
            pdf_name=d.get("pdf_name"),
            pdf_page=d.get("pdf_page"),
            section_name=d.get("section_name"),
            subsection_name=d.get("subsection_name"),
            chunk_type=ChunkType(d.get("chunk_type")),
            text=d.get("text"),
            keywords=d.get("keywords"),
        )


if __name__ == "__main__":
    chunk = Chunk(
        id="1",
        pdf_name="sample.pdf",
        pdf_page=1,
        section_name="Introduction",
        chunk_type=ChunkType.TEXT,
        text="This is a sample text chunk.",
        keywords=["sample", "text", "chunk"],
    )
    print(chunk)
