"""Main Streamlit application for RAG QA System."""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from config import get_settings
from contextual_document_processor import ContextualDocumentProcessor
from vector_store import VectorStoreManager
from query_engine import QueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG QA System", page_icon="📚", layout="wide", initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "settings" not in st.session_state:
        try:
            st.session_state.settings = get_settings()
        except Exception as e:
            st.error(f"Ошибка конфигурации: {e}")
            st.stop()

    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager(
            api_key=st.session_state.settings.pinecone_api_key,
            environment=st.session_state.settings.pinecone_environment,
            index_name=st.session_state.settings.pinecone_index_name,
            cohere_api_key=st.session_state.settings.cohere_api_key,
            reranker_top_n=st.session_state.settings.reranker_top_n,
        )

    if "document_processor" not in st.session_state:
        st.session_state.document_processor = ContextualDocumentProcessor(
            max_file_size_mb=st.session_state.settings.max_file_size_mb,
            max_files_count=st.session_state.settings.max_files_count,
            enable_contextual_extraction=st.session_state.settings.enable_contextual_extraction,
        )

    if "query_engine" not in st.session_state:
        st.session_state.query_engine = QueryEngine(
            vector_store_manager=st.session_state.vector_store_manager,
            enable_self_correction=st.session_state.settings.enable_self_correction,
            max_retry_attempts=st.session_state.settings.max_retry_attempts,
        )

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    if "index_stats" not in st.session_state:
        st.session_state.index_stats = {}


def render_sidebar():
    """Render sidebar with filters and settings."""
    st.sidebar.header("⚙️ Настройки поиска")

    # Search parameters
    similarity_top_k = st.sidebar.slider(
        "Количество результатов (этап 1)",
        min_value=10,
        max_value=50,
        value=st.session_state.settings.initial_retrieval_k,
        help="Количество документов для начального поиска (затем reranker отбирает лучшие)",
    )

    similarity_threshold = st.sidebar.slider(
        "Порог релевантности",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings.similarity_threshold,
        step=0.1,
        help="Минимальная релевантность для отображения результатов",
    )

    Advanced search filters
    st.sidebar.header("🔍 Расширенные фильтры")

    Basic filters
    author_filter = st.sidebar.text_input("Автор документа", placeholder="Введите имя автора...")

    Scientific content filters
    st.sidebar.subheader("📚 Научное содержание")

    has_equations = st.sidebar.checkbox("Содержит формулы/уравнения")
    has_methodology = st.sidebar.checkbox("Содержит методологию")
    has_results = st.sidebar.checkbox("Содержит результаты")
    has_abstract = st.sidebar.checkbox("Содержит аннотацию")

    Content type filters
    st.sidebar.subheader("📊 Тип содержания")

    content_types = st.sidebar.multiselect(
        "Выберите типы содержания",
        ["has_tables", "has_figures", "has_code"],
        format_func=lambda x: {
            "has_tables": "Таблицы",
            "has_figures": "Рисунки",
            "has_code": "Код",
        }.get(x, x),
    )

    Complexity filter
    complexity_filter = st.sidebar.selectbox(
        "Уровень сложности",
        ["Любой", "basic", "intermediate", "advanced"],
        format_func=lambda x: {
            "Любой": "Любой",
            "basic": "Базовый",
            "intermediate": "Средний",
            "advanced": "Продвинутый",
        }.get(x, x),
    )

    Clear filters button
    if st.sidebar.button("🗑️ Очистить фильтры"):
        st.rerun()

    Build metadata filters
    metadata_filters = {}

    if author_filter:
        metadata_filters["author"] = author_filter

    if has_equations:
        metadata_filters["has_equations"] = True

    if has_methodology:
        metadata_filters["has_methodology"] = True

    if has_results:
        metadata_filters["has_results"] = True

    if has_abstract:
        metadata_filters["has_abstract"] = True

    for content_type in content_types:
        metadata_filters[content_type] = True

    if complexity_filter != "Любой":
        metadata_filters["complexity_level"] = complexity_filter

    Store filters in session state
    st.session_state.search_params = {
        "similarity_top_k": similarity_top_k,
        "similarity_threshold": similarity_threshold,
        "metadata_filters": metadata_filters,
    }

    Index statistics
    if st.session_state.index_stats:
        st.sidebar.header("📊 Статистика индекса")
        stats = st.session_state.index_stats
        st.sidebar.metric("Всего векторов", stats.get("total_vectors", 0))
        st.sidebar.metric("Размерность", stats.get("dimension", 0))
        if stats.get("index_fullness"):
            st.sidebar.metric("Заполненность", f"{stats['index_fullness']:.1%}")

    Query statistics
    query_stats = st.session_state.query_engine.get_statistics()
    if query_stats["total_queries"] > 0:
        st.sidebar.header("📈 Статистика запросов")
        st.sidebar.metric("Всего запросов", query_stats["total_queries"])
        st.sidebar.metric("Успешные запросы", query_stats["successful_queries"])

        if query_stats.get("self_corrected_queries", 0) > 0:
            st.sidebar.metric("Самокоррекция", f"{query_stats['correction_rate']:.1f}%")
            st.sidebar.metric("Среднее число попыток", query_stats["avg_retry_count"])

        if query_stats["language_distribution"]:
            st.sidebar.write("**Языки запросов:**")
            for lang, count in query_stats["language_distribution"].items():
                st.sidebar.write(f"- {lang}: {count}")

    Enhanced features status
    st.sidebar.header("🚀 Активные улучшения")

    if st.session_state.settings.enable_self_correction:
        st.sidebar.success("✅ Самокоррекция включена")
    else:
        st.sidebar.warning("❌ Самокоррекция отключена")

    if st.session_state.vector_store_manager.reranker:
        st.sidebar.success("✅ Cohere Reranker активен")
    else:
        st.sidebar.warning("❌ Reranker недоступен")

    if st.session_state.settings.enable_contextual_extraction:
        st.sidebar.success("✅ Контекстная обработка включена")
    else:
        st.sidebar.warning("❌ Контекстная обработка отключена")


def render_document_upload():
    """Render document upload section."""
    st.header("📄 Загрузка документов")

    uploaded_files = st.file_uploader(
        "Выберите PDF файлы",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Максимум {st.session_state.settings.max_files_count} файлов, до {st.session_state.settings.max_file_size_mb}MB каждый",
    )

    if uploaded_files:
        st.write(f"**Выбрано файлов:** {len(uploaded_files)}")

        Show file details
        for file in uploaded_files:
            file_size_mb = file.size / (1024 * 1024)
            st.write(f"- {file.name} ({file_size_mb:.1f}MB)")

        if st.button("🚀 Обработать документы", type="primary"):
            with st.spinner("Обработка документов..."):
                Process documents
                documents = st.session_state.document_processor.process_uploaded_files(
                    uploaded_files
                )

                if documents:
                    st.success(f"Успешно обработано {len(documents)} документов")

                    Show extraction stats
                    if hasattr(st.session_state.document_processor, "get_extraction_stats"):
                        stats = st.session_state.document_processor.get_extraction_stats()
                        if stats["contextual_extraction_enabled"]:
                            st.info(
                                f"🧠 Контекстная обработка: {stats['available_extractors']} экстракторов активно"
                            )

                    Initialize or create index
                    with st.spinner("Создание векторного индекса..."):
                        if not st.session_state.vector_store_manager.create_index():
                            st.error("Ошибка создания индекса Pinecone")
                            return

                        if not st.session_state.vector_store_manager.create_vector_index(documents):
                            st.error("Ошибка создания векторного индекса")
                            return

                    st.session_state.documents_loaded = True
                    st.session_state.index_stats = (
                        st.session_state.vector_store_manager.get_index_stats()
                    )

                    Show success with feature info
                    success_msg = "Документы успешно проиндексированы!"
                    if st.session_state.vector_store_manager.reranker:
                        success_msg += " 🎯 Индекс готов к работе!"
                    st.success(success_msg)
                    st.rerun()
                else:
                    st.error("Не удалось обработать документы")


def render_query_interface():
    """Render query interface."""
    st.header("🤖 Поиск и вопросы")

    if not st.session_state.documents_loaded:
        Try to load existing index
        if st.session_state.vector_store_manager.load_existing_index():
            st.session_state.documents_loaded = True
            st.session_state.index_stats = st.session_state.vector_store_manager.get_index_stats()
        else:
            st.warning("Сначала загрузите документы для индексации")
            return

    Query input
    query = st.text_area(
        "Задайте вопрос по документам:",
        placeholder="Например: Что говорится о машинном обучении в документах?",
        height=100,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        search_button = st.button("🔍 Найти ответ", type="primary", disabled=not query.strip())

    with col2:
        if st.button("🗑️ Очистить историю"):
            st.session_state.query_engine.clear_history()
            st.success("История очищена")
            st.rerun()

    if search_button and query.strip():
        with st.spinner("Поиск ответа..."):
            result = st.session_state.query_engine.process_query(
                query=query, **st.session_state.search_params
            )

        if result["success"]:
            Display answer
            st.subheader("💡 Ответ")
            st.write(result["answer"])

            Display self-correction info
            if result.get("self_corrected", False):
                st.success(
                    f"🔄 Ответ улучшен через самокоррекцию (попыток: {result.get('retry_count', 0)})"
                )

            Display processing info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 Источников найдено: {len(result['sources'])}")
            with col2:
                if result.get("self_corrected", False):
                    st.info(f"🎯 Самокоррекция применена")
                else:
                    st.info(f"✅ Ответ сгенерирован с первой попытки")

            Display sources
            if result["sources"]:
                st.subheader("📚 Источники")
                for source in result["sources"]:
                    with st.expander(
                        f"📄 {source['filename']} (релевантность: {source['score']:.2f})"
                    ):
                        st.write(f"**Заголовок:** {source['title']}")
                        st.write(f"**Автор:** {source['author']}")
                        st.write(f"**Фрагмент текста:**")
                        st.write(source["text_snippet"])
        else:
            st.error(f"Ошибка: {result.get('error', 'Неизвестная ошибка')}")


def render_query_history():
    """Render query history."""
    st.header("📋 История запросов")

    history = st.session_state.query_engine.get_query_history()

    if not history:
        st.info("История запросов пуста")
        return

    for i, item in enumerate(reversed(history[-5:])):  # Show last 5
        with st.expander(f"❓ {item['query'][:50]}... ({item['timestamp'][:19]})"):
            st.write(f"**Вопрос:** {item['query']}")
            st.write(f"**Ответ:** {item['answer']}")
            st.write(f"**Язык:** {item['language']}")
            st.write(f"**Источников:** {len(item['sources'])}")
            if item.get("self_corrected", False):
                st.write(f"**Самокоррекция:** ✅ (попыток: {item.get('retry_count', 0)})")
            else:
                st.write(f"**Самокоррекция:** ❌")


def main():
    """Main application function."""
    st.title("📚 RAG QA System")
    st.markdown("Система вопросов и ответов для исследовательских команд")

    Initialize session state
    initialize_session_state()

    Render sidebar
    render_sidebar()

    Main content
    tab1, tab2, tab3 = st.tabs(["📄 Документы", "🔍 Поиск", "📋 История"])

    with tab1:
        render_document_upload()

    with tab2:
        render_query_interface()

    with tab3:
        render_query_history()


if __name__ == "__main__":
    main()
