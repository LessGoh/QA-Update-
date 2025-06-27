"""Query engine for semantic search and answer generation."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from langdetect import detect
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator

logger = logging.getLogger(__name__)


class QueryEngine:
    """Handle query processing and response generation."""

    def __init__(self, vector_store_manager, enable_self_correction=True, max_retry_attempts=2):
        self.vector_store_manager = vector_store_manager
        self.query_history = []
        self.enable_self_correction = enable_self_correction
        self.max_retry_attempts = max_retry_attempts
        self.retry_engine = None

    def detect_query_language(self, query: str) -> str:
        """Detect the language of the query."""
        try:
            return detect(query)
        except:
            return "en"

    def get_language_prompt(self, language: str) -> str:
        """Get system prompt based on detected language."""
        prompts = {
            "ru": """Ты помощник для ответов на вопросы на основе предоставленных документов.
            
Инструкции:
1. Отвечай на русском языке
2. Используй только информацию из предоставленных документов
3. Если информации недостаточно, так и скажи
4. Указывай источники своих ответов
5. Будь конкретным и точным""",
            "en": """You are an assistant for answering questions based on provided documents.
            
Instructions:
1. Answer in English
2. Use only information from the provided documents
3. If information is insufficient, say so
4. Cite your sources
5. Be specific and accurate""",
        }
        return prompts.get(language, prompts["en"])

    def _initialize_retry_engine(self, base_query_engine):
        """Initialize the retry query engine for self-correction."""
        if not self.enable_self_correction or not base_query_engine:
            return base_query_engine

        try:
            # Create evaluator for relevancy checking
            evaluator = RelevancyEvaluator(llm=self.vector_store_manager.llm)

            # Initialize retry engine
            retry_engine = RetryQueryEngine(
                query_engine=base_query_engine,
                evaluator=evaluator,
                max_retries=self.max_retry_attempts,
            )

            logger.info(
                f"Initialized self-correcting engine with {self.max_retry_attempts} max retries"
            )
            return retry_engine

        except Exception as e:
            logger.warning(f"Failed to initialize retry engine: {e}. Using base engine.")
            return base_query_engine

    def format_sources(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        """Format source nodes for display."""
        sources = []
        for i, node in enumerate(source_nodes):
            source_info = {
                "index": i + 1,
                "score": getattr(node, "score", 0.0),
                "filename": node.metadata.get("filename", "Unknown"),
                "title": node.metadata.get("title", "Unknown"),
                "author": node.metadata.get("author", "Unknown"),
                "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            }
            sources.append(source_info)
        return sources

    def process_query(
        self,
        query: str,
        similarity_top_k: int = 20,  # Increased for better initial retrieval
        similarity_threshold: float = 0.7,  # Will be removed when reranker is implemented
        metadata_filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process a query using self-correcting engine."""

        # Detect query language
        query_language = self.detect_query_language(query)

        # Get base query engine
        base_query_engine = self.vector_store_manager.get_query_engine(
            similarity_top_k=similarity_top_k, filters=metadata_filters
        )

        if not base_query_engine:
            return {
                "success": False,
                "error": "Query engine not available. Please upload documents first.",
            }

        # Initialize retry engine for self-correction
        query_engine = self._initialize_retry_engine(base_query_engine)

        try:
            # Execute query with potential self-correction
            response = query_engine.query(query)

            # Check if self-correction was applied
            self_corrected = hasattr(response, "retry_count") and response.retry_count > 0

            # With Cohere reranker, sources are already filtered and ranked by relevance
            # No need for similarity_threshold filtering
            if not response.source_nodes:
                return {
                    "success": True,
                    "answer": (
                        "Не найдено релевантных документов для ответа на ваш вопрос."
                        if query_language == "ru"
                        else "No relevant documents found to answer your question."
                    ),
                    "sources": [],
                    "language": query_language,
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "self_corrected": self_corrected,
                }

            # Format sources (already reranked by Cohere)
            sources = self.format_sources(response.source_nodes)

            # Prepare result
            result = {
                "success": True,
                "answer": response.response,
                "sources": sources,
                "language": query_language,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "self_corrected": self_corrected,
                "retry_count": getattr(response, "retry_count", 0),
            }

            # Add to history
            self.query_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Error in self-correcting query: {e}")
            return {"success": False, "error": f"Error processing query: {str(e)}"}

    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history."""
        return self.query_history[-limit:] if self.query_history else []

    def clear_history(self):
        """Clear query history."""
        self.query_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get query statistics including self-correction metrics."""
        if not self.query_history:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "language_distribution": {},
                "recent_queries": 0,
                "self_corrected_queries": 0,
                "avg_retry_count": 0.0,
            }

        successful_queries = [q for q in self.query_history if q.get("success", False)]

        # Language distribution
        languages = [q.get("language", "unknown") for q in successful_queries]
        lang_dist = {}
        for lang in languages:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        # Self-correction statistics
        corrected_queries = [q for q in successful_queries if q.get("self_corrected", False)]
        retry_counts = [q.get("retry_count", 0) for q in successful_queries]
        avg_retry_count = sum(retry_counts) / len(retry_counts) if retry_counts else 0

        return {
            "total_queries": len(self.query_history),
            "successful_queries": len(successful_queries),
            "language_distribution": lang_dist,
            "recent_queries": len([q for q in self.query_history[-24:] if q.get("success", False)]),
            "self_corrected_queries": len(corrected_queries),
            "correction_rate": (
                round(len(corrected_queries) / len(successful_queries) * 100, 1)
                if successful_queries
                else 0.0
            ),
            "avg_retry_count": round(avg_retry_count, 2),
        }
