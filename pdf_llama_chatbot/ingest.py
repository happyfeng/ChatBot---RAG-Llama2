# -*- coding: utf-8 -*-
"""
Ingest PDFs: load, chunk, embed, and persist vector store for RAG.
Run once (or when PDFs change): python ingest.py
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

def get_documents(pdf_dir: Path = None):
    pdf_dir = pdf_dir or config.PDF_DIR
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created empty folder: {pdf_dir}. Add PDF files there and run again.")
        return []
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDFs in {pdf_dir}")
    return docs

def split_documents(docs, chunk_size=1024, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_and_persist_vector_store(chunks=None, persist_dir: str = None):
    config.ensure_dirs()
    if chunks is None:
        docs = get_documents()
        if not docs:
            return None
        chunks = split_documents(docs)
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vector_store.persist()
    print(f"Vector store saved to {persist_dir}")
    return vector_store

if __name__ == "__main__":
    build_and_persist_vector_store()
