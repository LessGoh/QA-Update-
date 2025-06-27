"""Contextual document processor with metadata extraction for enhanced search."""

import logging
from typing import List, Dict, Any, Optional
import streamlit as st
from llama_index.core import Document
from llama_index.core.extractors import (
    SummaryExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SentenceSplitter
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class ContextualDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with contextual metadata extraction."""

    def __init__(
        self,
        max_file_size_mb: int = 50,
        max_files_count: int = 100,
        enable_contextual_extraction: bool = True,
    ):
        super().__init__(max_file_size_mb, max_files_count)

        self.enable_contextual_extraction = enable_contextual_extraction

        # Replace simple SentenceSplitter with same settings but ready for extractors
        self.node_parser = SentenceSplitter(
            chunk_size=1024, chunk_overlap=20  # Keep existing settings
        )

        # Initialize contextual extractors
        self.extractors = []
        if self.enable_contextual_extraction:
            try:
                self.extractors = [
                    SummaryExtractor(
                        summaries=["self"],  # Brief context summary for chunk
                        llm=None,  # Will use default LLM from Settings
                        show_progress=False,
                    ),
                    KeywordExtractor(
                        keywords=10, llm=None, show_progress=False  # Top-10 keywords for chunk
                    ),
                    QuestionsAnsweredExtractor(
                        questions=3,  # 3 potential questions for chunk
                        llm=None,
                        show_progress=False,
                    ),
                ]
                logger.info(f"Initialized {len(self.extractors)} contextual extractors")
            except Exception as e:
                logger.warning(f"Failed to initialize extractors: {e}")
                self.extractors = []
                self.enable_contextual_extraction = False

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Create contextual chunks with metadata enhancement."""
        # First create regular nodes (existing logic)
        nodes = []
        for document in documents:
            doc_nodes = self.node_parser.get_nodes_from_documents([document])
            nodes.extend(doc_nodes)

        # If contextual extraction is disabled, return regular nodes
        if not self.enable_contextual_extraction or not self.extractors:
            logger.info(f"Created {len(nodes)} regular nodes (contextual extraction disabled)")
            return nodes

        # Add contextual metadata through extractors
        try:
            original_count = len(nodes)
            original_nodes = nodes.copy()  # Backup for fallback

            # Summary extractor - adds contextual summaries
            if len(self.extractors) > 0:
                nodes = self.extractors[0].extract(nodes)
                logger.debug("Applied summary extractor")

            # Keyword extractor - adds keywords
            if len(self.extractors) > 1:
                nodes = self.extractors[1].extract(nodes)
                logger.debug("Applied keyword extractor")

            # Questions extractor - adds potential questions
            if len(self.extractors) > 2:
                nodes = self.extractors[2].extract(nodes)
                logger.debug("Applied questions extractor")

            logger.info(f"Enhanced {original_count} nodes with contextual metadata")

        except Exception as e:
            logger.warning(f"Failed to extract contextual metadata: {e}")
            # Fallback to original nodes if extractors fail
            nodes = original_nodes if "original_nodes" in locals() else nodes
            logger.info("Falling back to regular nodes without contextual enhancement")

        return nodes

    def process_uploaded_files(self, uploaded_files: List[Any]) -> List[Any]:
        """Process files using contextual chunking."""
        if not self.validate_files(uploaded_files):
            return []

        documents = []
        progress_bar = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Extract text and metadata (existing logic)
                pdf_content = uploaded_file.read()
                text = self.extract_text_from_pdf(pdf_content)

                if not text.strip():
                    st.warning(f"Не удалось извлечь текст из {uploaded_file.name}")
                    continue

                # Create enhanced metadata (now includes scientific features)
                metadata = self.extract_pdf_metadata(pdf_content, uploaded_file.name)
                metadata["language"] = self.detect_language(text)
                metadata["file_size"] = uploaded_file.size

                # Create document
                document = Document(text=text, metadata=metadata)
                documents.append(document)

                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Ошибка обработки файла {uploaded_file.name}: {e}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.empty()
        return documents

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about contextual extraction."""
        return {
            "contextual_extraction_enabled": self.enable_contextual_extraction,
            "available_extractors": len(self.extractors),
            "extractor_types": (
                [type(e).__name__ for e in self.extractors] if self.extractors else []
            ),
        }
