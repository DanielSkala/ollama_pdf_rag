import base64
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import spacy
from dotenv import dotenv_values
from openai import OpenAI

from src.core.models import Chunk, ChunkType

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

ENV_VARS = dotenv_values("../../.env.local")
OPENAI_API_KEY = ENV_VARS["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"


class ImageTranscriber:
    """
    Transcribes images in a PDF document using the OpenAI API
    """

    def __init__(self, openai_api_key, model, cache_dir="image_transcriptions_cache"):
        self.OPENAI_API_KEY = openai_api_key
        self.MODEL = model
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def convert_images_in_page_to_text(self, document, page) -> str:
        """
        Processes the page by replacing qualifying images with their text transcriptions.
        It first checks if a transcription is cached; if not, it calls the OpenAI API.
        Returns the modified page.
        """
        logger.info(f"Replacing images by text. Page number: {page.number}")
        image_info = page.get_image_info()

        for im_index, image in enumerate(page.get_images()):
            if self._is_image_valid(image_info[im_index]):
                xref = image[0]  # XREF of the image is an integer.
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"].lower()

                if image_ext in ("jpg", "jpeg", "png"):
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    cache_file = os.path.join(self.cache_dir, f"{xref}.{image_ext}.txt")

                    if os.path.exists(cache_file):
                        with open(cache_file, "r", encoding="utf-8") as f:
                            logger.info(f"Using cached transcription for image {xref}")
                            image_text = f.read().strip()
                    else:
                        logger.info(
                            f"Using OpenAI API for transcription of image {xref}"
                        )
                        image_text = self._image_to_text(base64_image, image_ext)
                        with open(cache_file, "w", encoding="utf-8") as f:
                            f.write(image_text)

                    # Insert the transcribed text in place of the image.
                    bbox = page.get_image_rects(xref)[0]
                    page.insert_text(bbox[0:2], image_text, fontsize=1, color=(1, 0, 0))

        return page

    def _image_to_text(self, base64_image, ext: str) -> str:
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        max_retries = 10
        delay = 1  # initial delay in seconds

        image_prompt = (
            "You will be given an image from the MicroStep-MIS documentation. "
            "This image will likely be a screenshot, diagram, or a schema. "
            "Your task is to describe what is on this image in 2-3 brief sentences. "
            "If the image appears to be a logo/icon, or something generic, simply say that the image "
            "is a logo of MicroStep-MIS. Output the text description in this format: \n\n"
            "[image]: <insert your description here>\n\n"
            "Example: \n\n"
            "[image]: Schema depicting an airport runway equipped with various meteorological and "
            "environmental sensors..."
        )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": image_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{ext};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e  # re-raise the exception if out of retries
                time.sleep(delay)
                delay *= 2  # exponential backoff

    def _is_image_valid(self, image_info) -> bool:
        """
        Image is valid if it is large enough (to omit small icons) and if it is outside of header.

        Example or logo in header:
        'bbox': (74.3499984741211, 38.6999626159668, 241.5, 65.14996337890625)
        """

        min_image_size = 1000
        min_image_width = 50
        min_image_height = 50
        header_y_coordinate = 70  # y-coordinate of the header
        bbox = image_info["bbox"]

        im_s = image_info["size"]
        im_w = image_info["width"]
        im_h = image_info["height"]

        return (
            im_s >= min_image_size
            and im_w >= min_image_width
            and im_h >= min_image_height
            and bbox[3] > header_y_coordinate
        )


class PDFLoader:
    """
    Loads PDF documents using PyMuPDF (fitz) and returns the full text along
    with a list of (page_number, start_offset) pairs.
    """

    def __init__(self, image_transcriber: ImageTranscriber = None):
        self.image_transcriber = image_transcriber

    def load(self, file_path: Path) -> Tuple[str, List[Tuple[int, int]]]:
        try:
            logger.info(f"Loading PDF from {file_path}")
            doc = fitz.open(str(file_path))
            full_text = ""
            page_starts: List[Tuple[int, int]] = []
            current_offset = 0

            for page in doc:
                if self.image_transcriber:
                    page = self.image_transcriber.convert_images_in_page_to_text(
                        doc, page
                    )

                text = page.get_text()
                page_starts.append((page.number, current_offset))
                full_text += text
                current_offset += len(text)

            doc.close()
            return full_text, page_starts

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise


class ChunkingStrategy(ABC):
    """
    Abstract base class for different document chunking strategies.
    The chunk_document method returns a list of Chunk objects.
    """

    @abstractmethod
    def chunk_document(
        self, document_text: str, pdf_name: str, page_starts: List[Tuple[int, int]]
    ) -> List[Chunk]:
        pass

    @staticmethod
    def _get_page_for_offset(offset: int, page_starts: List[Tuple[int, int]]) -> int:
        """
        Determines the page number for a given character offset based on the
        list of (page_number, start_offset) pairs.
        """
        page_for_chunk = None
        for page_num, start_offset in page_starts:
            if offset >= start_offset:
                page_for_chunk = page_num
            else:
                break
        return page_for_chunk + 1 if page_for_chunk is not None else 0


class ConstantLengthChunkStrategy(ChunkingStrategy):
    """
    Dummy strategy that splits text into fixed-size chunks with optional overlap.
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, document_text: str, pdf_name: str, page_starts: List[Tuple[int, int]]
    ) -> List[Chunk]:
        chunks = []
        start = 0
        chunk_index = 1

        while start < len(document_text):
            end = start + self.chunk_size
            text_chunk = document_text[start:end].strip()
            pdf_page = self._get_page_for_offset(start, page_starts)
            chunk = Chunk(
                id=f"{pdf_name}-{chunk_index}",
                pdf_name=pdf_name,
                pdf_page=pdf_page,
                section_name=None,  # TODO: Implement section detection
                subsection_name=None,  # TODO: Implement subsection detection
                chunk_type=ChunkType.TEXT,
                text=text_chunk,
            )
            chunks.append(chunk)
            chunk_index += 1
            # Advance the start pointer to create overlapping chunks.
            start += self.chunk_size - self.chunk_overlap

        return chunks


class AdvancedParagraphChunkStrategy(ChunkingStrategy):
    """
    Splits document into (sub)sections. Each (sub)section is a separate chunk.
    This approach requires that the pdf file contains a table of contents.

    [I] _extract_section_names(): We extract, from up to down, the (sub)section names from the table of contents.
    The extraction using regex "^(?:([\d.]+)\s*)?\n?([A-Za-z][\w\s–-]+?)\s+\.{5,}\s*(\d+)":
                                            (a) (?:([\d.]+)\s*)?\n? - captures numbers like 3, 1.2 or 1.12 with or
                                                without whitespace. The symbol "?" behind the parentheses makes the
                                                numbers optional, which means that chapters without numbers like
                                                "Introduction" can also be extracted. \n? is optional newline.
                                            (b) ([A-Za-z][\w\s–-]+?) - section title with some extra flexibility.
                                            (c) \s+\.{5,}\s* - at least 10 dots between the section title and the page
                                                number.
                                            (d) (\d+) - page number.
    [II] _find_section_occurrences(): We look for the next occurrence next_{i+1} of extracted (sub)section i+1.
    To accept the next match, the following has to be satisfied:
                                            (a) next_{i+1} is preceded and followed by \n;
                                            (b) next_{i+1} is not a bullet point;
                                            (c) next_{i+1} comes after next_{i}.
    [III] chunk_document(): Parsing the pdf according to the starting/ending indices of each next_{i}
    [IV] _create_chunk(): Creates the chunk object.
    """

    def chunk_document(
        self, document_text: str, pdf_name: str, page_starts: List[Tuple[int, int]]
    ) -> List[Chunk]:
        sections = self._find_section_occurrences(document_text)

        sections.append(("END", len(document_text)))

        chunks = []
        for i, (start_sec, next_sec) in enumerate(zip(sections, sections[1:])):
            chunk = self._create_chunk(
                document_text=document_text,
                pdf_name=pdf_name,
                page_starts=page_starts,
                chunk_index=i + 1,
                start_section=start_sec,
                end_section=next_sec,
            )
            chunks.append(chunk)

        return chunks

    def _extract_content_words(self, text: str) -> List[str]:
        """
        Extract content words from text using spaCy.
        Only tokens with POS tags in {NOUN, PROPN, ADJ} are considered.
        Return their lower-case lemmas (including duplicates).
        """
        doc = nlp(text)
        valid_pos = {"NOUN", "PROPN", "ADJ"}
        return [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in valid_pos and token.is_alpha
        ]

    def _extract_section_names(self, document_text: str) -> List[str]:
        matches = re.findall(
            r"^(?:([\d.]+)\s*)?\n?([A-Za-z][\w\s–-]+?)\s+\.{5,}\s*(\d+)",
            document_text,
            re.MULTILINE,
        )
        return [match[1].strip().split("\n")[-1] for match in matches]

    def _find_section_occurrences(self, document_text: str) -> List[Tuple[str, int]]:
        section_names = self._extract_section_names(document_text)
        occurrences = []
        start_index = 0
        for name in section_names:
            pattern = rf"(?<!•\s)\n{re.escape(name)} \n"
            match = re.search(pattern, document_text[start_index:])
            if not match:
                print(f"Section not found: {name}")
                continue
            pos = start_index + match.start()
            occurrences.append((name, pos))
            start_index = pos
        return occurrences

    @staticmethod
    def _remove_footer(text: str) -> str:
        pattern = re.compile(
            r"(?:\n\s+)*"  # Arbitrary number of newline+whitespace sequences
            r"-\d+-\n"  # Page number
            r"(?:\n\s+)*"  # Arbitrary number of newline+whitespace sequences
            r".*?"  # Arbitrary number of characters, non-greedy
            r"(?:\n\s+)*"  # Arbitrary number of newline+whitespace sequences
            r"Web-site.*",  # Footer text
            re.DOTALL,
        )
        return re.sub(pattern, "", text)

    def _create_chunk(
        self,
        document_text: str,
        pdf_name: str,
        page_starts: List[Tuple[int, int]],
        chunk_index: int,
        start_section: Tuple[str, int],
        end_section: Tuple[str, int] = None,
    ) -> Chunk:
        section_name, start_index = start_section
        end_index = end_section[1] if end_section else None
        chunk_text = document_text[start_index:end_index].strip()

        if not chunk_text:
            return None

        pdf_page = self._get_page_for_offset(start_index, page_starts)
        return Chunk(
            id=f"{pdf_name}-{chunk_index}",
            pdf_name=pdf_name,
            pdf_page=pdf_page,
            section_name=section_name,
            subsection_name=None,
            chunk_type=ChunkType.TEXT,
            text=self._remove_footer(chunk_text),
            keywords=self._extract_content_words(chunk_text),
        )


class DocumentProcessor:
    """
    Loads and processes a PDF using a specified chunking strategy.
    """

    def __init__(self, loader: PDFLoader, chunk_strategy: ChunkingStrategy):
        self.loader = loader
        self.chunk_strategy = chunk_strategy

    def load_pdf(self, file_path: Path) -> Tuple[str, List[Tuple[int, int]]]:
        return self.loader.load(file_path)

    def process_pdf(self, file_path: Path) -> List[Chunk]:
        """
        Loads the PDF, splits the text into chunks using the chunking strategy,
        and returns a list of Chunk objects with the proper page numbers.
        """
        document_text, page_starts = self.load_pdf(file_path)
        pdf_name = file_path.name
        return self.chunk_strategy.chunk_document(document_text, pdf_name, page_starts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_transcriber = ImageTranscriber(openai_api_key=OPENAI_API_KEY, model=MODEL)

    # Automatically transcribe images when the PDFLoader gets the image_transcriber
    # If it is set to None, images will NOT be transcribed
    pdf_loader = PDFLoader(image_transcriber=image_transcriber)

    pdf_path = Path("../../data/pdfs/microstepexample.pdf")

    # Example 3: Use the advanced paragraph chunk strategy
    # Option A: chunk_i = (sub)section_i as extracted from the table of contents

    advanced_paragraph_chunk_strategy = AdvancedParagraphChunkStrategy()
    processor_paragraph = DocumentProcessor(
        loader=pdf_loader, chunk_strategy=advanced_paragraph_chunk_strategy
    )

    paragraph_chunks = processor_paragraph.process_pdf(pdf_path)
    print(f"Number of chunks (advanced paragraph strategy): {len(paragraph_chunks)}")

    for i, chunk in enumerate(paragraph_chunks):
        print(f"--- Chunk {i + 1} ---")
        print(chunk)
        print("\n")
