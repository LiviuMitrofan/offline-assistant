"""Directory scanning, document loading, and vector store management."""

import os
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from .config import CHROMA_DIR
from .file_utils import (
    get_file_metadata,
    extract_content,
    generate_document_summary,
    extract_document_keywords,
)

console = Console()

# File types that benefit from LLM-generated summaries and keyword extraction.
_SUMMARISABLE_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx",
    ".py", ".js", ".ts", ".html", ".csv", ".json", ".yaml", ".yml",
}
# Image extensions — their "content" IS already an LLM description, so we
# skip a second LLM call and reuse the description directly.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def scan_directories(directories: list[str], ollama_model: str = "") -> list[Document]:
    """
    Walk configured directories and produce LangChain Documents.

    For every file:
      • A *content* document  — metadata header + extracted text (always).
      • A *summary* document  — LLM-written 2-3 sentence summary + keywords
                                (when ollama_model is provided and the file
                                has extractable text).

    After scanning each directory a *directory overview* document is created
    listing every file found in that folder, giving the agent a fast way to
    answer "what's in the <folder> folder?" questions.
    """
    documents: list[Document] = []
    total_files = 0
    indexed_content = 0
    indexed_metadata_only = 0
    summaries_generated = 0

    # dir_path → list of lightweight file dicts for the overview document
    dir_file_map: dict[str, list[dict]] = defaultdict(list)

    enrich = bool(ollama_model)
    if enrich:
        console.print(f"  [dim]LLM enrichment enabled — summaries & keywords via '{ollama_model}'[/]")
    else:
        console.print("  [dim]LLM enrichment disabled (no ollama_model provided)[/]")

    for directory in directories:
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            console.print(f"  [yellow]Warning:[/] directory does not exist, skipping: {directory}")
            continue

        console.print(f"  Scanning: [cyan]{directory}[/]")
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                filepath = os.path.join(root, fname)
                total_files += 1

                try:
                    metadata = get_file_metadata(filepath)
                except (OSError, PermissionError):
                    continue

                ext = metadata["extension"]

                # --- Content extraction ---
                content = extract_content(filepath)
                has_content = bool(content and content.strip())

                # --- LLM enrichment: summary + keywords ---
                summary: str | None = None
                keywords: list[str] = []

                if enrich and has_content:
                    if ext in _IMAGE_EXTENSIONS:
                        # Image content IS the LLM description — reuse it directly.
                        summary = content.strip()[:500]
                        keywords = extract_document_keywords(content, ollama_model)
                    elif ext in _SUMMARISABLE_EXTENSIONS:
                        console.print(f"    [dim]Enriching {fname}…[/]")
                        summary = generate_document_summary(content, ollama_model)
                        keywords = extract_document_keywords(content, ollama_model)

                    if summary or keywords:
                        summaries_generated += 1

                # Persist summary and keywords into every content chunk's metadata
                # (the text splitter propagates metadata to all child chunks).
                metadata["summary"] = summary or ""
                metadata["keywords"] = ", ".join(keywords)
                metadata["chunk_type"] = "content"

                # --- Build the content document ---
                meta_summary = (
                    f"FILENAME: {metadata['filename']}\n"
                    f"PATH: {metadata['source']}\n"
                    f"TYPE: {ext or 'no extension'}\n"
                    f"SIZE: {metadata['size_bytes']} bytes\n"
                    f"MODIFIED: {metadata['modified']}\n"
                    f"CREATED: {metadata['created']}\n"
                    f"DIRECTORY: {metadata['directory']}"
                )
                if summary:
                    meta_summary += f"\nSUMMARY: {summary}"
                if keywords:
                    meta_summary += f"\nKEYWORDS: {metadata['keywords']}"

                if has_content:
                    page_content = f"{meta_summary}\n\nContent:\n{content}"
                    indexed_content += 1
                else:
                    page_content = meta_summary
                    indexed_metadata_only += 1

                documents.append(Document(page_content=page_content, metadata=metadata))

                # --- Build a separate, focused summary document ---
                # This document is short (never split), embeds with high
                # semantic density, and makes broad queries like "what AI docs
                # do I have?" surface the right file immediately.
                if summary:
                    summary_meta = {**metadata, "chunk_type": "summary"}
                    summary_page = (
                        f"SUMMARY OF: {metadata['filename']}\n"
                        f"PATH: {metadata['source']}\n"
                        f"TYPE: {ext}\n"
                        f"SUMMARY: {summary}"
                    )
                    if keywords:
                        summary_page += f"\nKEYWORDS: {metadata['keywords']}"
                    documents.append(Document(page_content=summary_page, metadata=summary_meta))

                # --- Track for directory overview ---
                dir_file_map[root].append({
                    "filename": metadata["filename"],
                    "extension": ext or "?",
                    "size_bytes": metadata["size_bytes"],
                    "modified": metadata["modified"],
                    "summary": summary or "",
                })

    # --- Create one directory overview document per folder ---
    for dir_path, file_list in dir_file_map.items():
        lines = [f"DIRECTORY OVERVIEW: {dir_path}", f"TOTAL FILES: {len(file_list)}", "FILES:"]
        for f in file_list:
            size_kb = f["size_bytes"] / 1024
            line = f"  - {f['filename']}  ({f['extension']}, {size_kb:.1f} KB, modified {f['modified'][:10]})"
            if f["summary"]:
                line += f"\n      Summary: {f['summary'][:150]}"
            lines.append(line)

        overview_content = "\n".join(lines)
        overview_meta = {
            "source": dir_path,
            "filename": os.path.basename(dir_path),
            "extension": "",
            "size_bytes": 0,
            "modified": "",
            "created": "",
            "directory": str(os.path.dirname(dir_path)),
            "summary": "",
            "keywords": "",
            "chunk_type": "directory_overview",
        }
        documents.append(Document(page_content=overview_content, metadata=overview_meta))

    console.print(
        f"\n  Scan complete: [bold]{total_files}[/] files found, "
        f"[green]{indexed_content}[/] with content, "
        f"[dim]{indexed_metadata_only}[/] metadata-only, "
        f"[cyan]{summaries_generated}[/] enriched with summaries/keywords, "
        f"[yellow]{len(dir_file_map)}[/] directory overview(s) created"
    )
    return documents


def build_vectorstore(
    documents: list[Document],
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Chroma:
    """Split documents, embed them, and store in Chroma."""
    console.print("\n  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    console.print(f"  Created [bold]{len(chunks)}[/] chunks from [bold]{len(documents)}[/] documents")

    console.print(f"\n  Generating embeddings with [cyan]'{embedding_model}'[/] (this may take a while)...")
    embeddings = OllamaEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="files",
    )
    console.print(f"  Vector store saved to [dim]{CHROMA_DIR}[/]")
    return vectorstore


def load_existing_vectorstore(embedding_model: str) -> Chroma | None:
    """Load an existing Chroma vector store if it exists."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        embeddings = OllamaEmbeddings(model=embedding_model)
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="files",
        )
    return None


def index_directory(
    vectorstore: Chroma,
    directory: str,
    ollama_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """
    Scan a single new directory and add its documents to an existing vectorstore.

    This is an incremental operation — existing documents are untouched.

    Returns:
        Number of chunks added.
    """
    documents = scan_directories([directory], ollama_model=ollama_model)
    if not documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)
    console.print(f"  Added [bold]{len(chunks)}[/] chunks from [cyan]{directory}[/]")
    return len(chunks)


def remove_directory_from_store(vectorstore: Chroma, directory: str) -> int:
    """
    Delete all indexed documents whose source path is under *directory*.

    Works by fetching all document IDs from Chroma and filtering in Python
    (Chroma does not support prefix/startswith filters natively).

    Returns:
        Number of chunks deleted.
    """
    directory = os.path.normpath(directory)
    try:
        result = vectorstore._collection.get(include=["metadatas"])
    except Exception as e:
        console.print(f"  [red]Could not fetch documents for removal: {e}[/]")
        return 0

    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(result["ids"], result["metadatas"])
        if os.path.normpath(meta.get("source", "")).startswith(directory)
        or os.path.normpath(meta.get("directory", "")).startswith(directory)
    ]

    if ids_to_delete:
        vectorstore._collection.delete(ids=ids_to_delete)
        console.print(f"  Removed [bold]{len(ids_to_delete)}[/] chunks for [cyan]{directory}[/]")

    return len(ids_to_delete)
