"""Document processing module for PDF handling."""

import os
import logging
import re
from typing import List, Dict, Any, Optional
from io import BytesIO
import PyPDF2
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from langdetect import detect
import streamlit as st

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle PDF document processing and text extraction."""

    def __init__(self, max_file_size_mb: int = 50, max_files_count: int = 100):
        self.max_file_size_mb = max_file_size_mb
        self.max_files_count = max_files_count
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    def validate_files(self, uploaded_files: List[Any]) -> bool:
        """Validate uploaded files against size and count limits."""
        if len(uploaded_files) > self.max_files_count:
            st.error(f"Слишком много файлов: {len(uploaded_files)} > {self.max_files_count}")
            return False

        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                st.error(
                    f"Файл {file.name} превышает максимальный размер: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"
                )
                return False

        return True

    def extract_pdf_metadata(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Extract enhanced metadata from PDF content including scientific features."""
        metadata = {
            "title": filename,
            "author": "Unknown",
            "creation_date": None,
            "filename": filename,
        }

        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            if pdf_reader.metadata:
                metadata.update(
                    {
                        "title": pdf_reader.metadata.get("/Title", filename),
                        "author": pdf_reader.metadata.get("/Author", "Unknown"),
                        "creation_date": pdf_reader.metadata.get("/CreationDate", None),
                    }
                )

            # Extract full text for content analysis
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text()

            # Add scientific metadata
            metadata.update(self._extract_research_metadata(full_text))

        except Exception as e:
            logger.warning(f"Could not extract metadata from {filename}: {e}")

        return metadata

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            return detect(text[:1000])  # Use first 1000 chars for detection
        except:
            return "en"  # Default to English

    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[Document]:
        """Process uploaded files and return LlamaIndex documents."""
        if not self.validate_files(uploaded_files):
            return []

        documents = []
        progress_bar = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Read file content
                pdf_content = uploaded_file.read()

                # Extract text
                text = self.extract_text_from_pdf(pdf_content)
                if not text.strip():
                    st.warning(f"Не удалось извлечь текст из {uploaded_file.name}")
                    continue

                # Extract metadata
                metadata = self.extract_pdf_metadata(pdf_content, uploaded_file.name)
                metadata["language"] = self.detect_language(text)
                metadata["file_size"] = uploaded_file.size

                # Create LlamaIndex document
                document = Document(text=text, metadata=metadata)
                documents.append(document)

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Ошибка обработки файла {uploaded_file.name}: {e}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.empty()
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Any]:
        """Split documents into chunks using LlamaIndex node parser."""
        nodes = []
        for document in documents:
            doc_nodes = self.node_parser.get_nodes_from_documents([document])
            nodes.extend(doc_nodes)
        return nodes

    def _extract_research_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata for scientific documents."""
        text_lower = text.lower()

        return {
            # Mathematical content - KEY for formula search
            "has_equations": bool(re.search(r"[∑∇∈αβγσμλπδ]|\\[a-zA-Z]+|\$.*\$", text)),
            "has_mathematical_notation": bool(re.search(r"[∀∃⟨⟩∧∨¬→↔]", text)),
            # Scientific paper structure - for section navigation
            "has_abstract": "abstract" in text_lower,
            "has_introduction": "introduction" in text_lower,
            "has_methodology": any(
                word in text_lower for word in ["methodology", "method", "approach", "algorithm"]
            ),
            "has_results": "results" in text_lower,
            "has_conclusion": any(
                word in text_lower for word in ["conclusion", "summary", "discussion"]
            ),
            "has_references": any(
                word in text_lower for word in ["references", "bibliography", "cited"]
            ),
            # Content elements - for quality filtering
            "has_tables": bool(re.search(r"\btable\s+\d+|\btab\.\s*\d+", text_lower)),
            "has_figures": bool(re.search(r"\bfig(?:ure)?\s+\d+|\bfig\.\s*\d+", text_lower)),
            "has_code": bool(re.search(r"```|def\s+\w+|class\s+\w+|import\s+\w+", text)),
            # Basic classification - for group filters
            "keywords": self._extract_simple_keywords(text),
            "document_sections": self._detect_sections(text_lower),
            "page_count": len(text.split("\n\n")),  # Approximate estimation
            "text_length": len(text),
            # Simple complexity assessment - for audience
            "complexity_level": self._assess_complexity_simple(text),
            "estimated_reading_time": max(1, len(text.split()) // 200),  # words per minute
        }

    def _extract_simple_keywords(self, text: str, max_keywords: int = 15) -> List[str]:
        """Simple keyword extraction without external libraries."""
        # Remove stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "и",
            "в",
            "на",
            "с",
            "по",
            "для",
            "от",
            "до",
            "как",
            "что",
            "это",
            "или",
            "но",
        }

        words = re.findall(r"\b[a-zA-Zа-яА-Я]{4,}\b", text.lower())
        word_freq = {}

        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top-N by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]

    def _detect_sections(self, text_lower: str) -> List[str]:
        """Determine document sections."""
        sections = []
        section_patterns = {
            "abstract": r"\babstract\b",
            "introduction": r"\bintroduction\b",
            "methodology": r"\b(?:methodology|methods?)\b",
            "results": r"\bresults?\b",
            "discussion": r"\bdiscussion\b",
            "conclusion": r"\bconclusions?\b",
            "references": r"\b(?:references|bibliography)\b",
        }

        for section, pattern in section_patterns.items():
            if re.search(pattern, text_lower):
                sections.append(section)

        return sections

    def _assess_complexity_simple(self, text: str) -> str:
        """Simple complexity assessment based on heuristics."""
        # Count technical terms and long sentences
        sentences = re.split(r"[.!?]+", text)
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )

        technical_patterns = [
            r"\b(?:algorithm|neural|machine learning|deep learning|optimization)\b",
            r"\b(?:regression|classification|clustering|validation)\b",
            r"\b(?:statistical|probability|distribution|correlation)\b",
        ]

        technical_score = sum(
            len(re.findall(pattern, text.lower())) for pattern in technical_patterns
        )

        if technical_score > 20 or avg_sentence_length > 25:
            return "advanced"
        elif technical_score > 10 or avg_sentence_length > 20:
            return "intermediate"
        else:
            return "basic"
