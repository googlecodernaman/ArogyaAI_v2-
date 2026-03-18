"""
MEDORBY Backend — RAG Module
Contains FAISS-based retrieval engine and medical report processor.
"""
from .engine import get_rag_engine, rebuild_rag_engine, RAGEngine
from .report_processor import process_report, get_all_reports, delete_report, get_report_text

__all__ = [
    "get_rag_engine", "rebuild_rag_engine", "RAGEngine",
    "process_report", "get_all_reports", "delete_report", "get_report_text",
]
