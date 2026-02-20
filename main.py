"""
Offline File System Assistant — CLI entry point.

Usage:
    python main.py
"""

import shutil
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

from src.config import load_config, CHROMA_DIR
from src.indexing import scan_directories, build_vectorstore, load_existing_vectorstore
from src.chain import build_chain

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Offline File System Assistant[/]\n"
            "[dim]Powered by Ollama + LangChain[/]",
            border_style="cyan",
        )
    )


def main():
    print_banner()

    # Load config
    config = load_config()
    directories = config.get("directories", [])
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 200)
    ollama_model = config.get("ollama_model", "qwen3:8b")
    embedding_model = config.get("embedding_model", "nomic-embed-text")
    retrieval_k = config.get("retrieval_k", 10)

    if not directories:
        console.print("[red]Error:[/] No directories configured in config.yaml")
        sys.exit(1)

    console.print(f"\n[bold]Model:[/]           {ollama_model}")
    console.print(f"[bold]Embedding model:[/] {embedding_model}")
    console.print(f"[bold]Directories:[/]      {len(directories)}")
    for d in directories:
        console.print(f"  [dim]>[/] {d}")

    # Check if we should re-index or use existing store
    vectorstore = None
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        reindex = Confirm.ask("\nExisting index found. Re-index?", default=False)
        if not reindex:
            console.print("\nLoading existing vector store...")
            vectorstore = load_existing_vectorstore(embedding_model)
            if vectorstore:
                console.print("  [green]Loaded successfully.[/]")
        else:
            shutil.rmtree(CHROMA_DIR)
            console.print("  [dim]Cleared existing index.[/]")

    if vectorstore is None:
        console.print("\n[bold]Indexing files...[/]")
        start = time.time()
        documents = scan_directories(directories, ollama_model=ollama_model)
        if not documents:
            console.print("[red]No files found in the configured directories.[/]")
            sys.exit(1)

        vectorstore = build_vectorstore(
            documents, embedding_model, chunk_size, chunk_overlap
        )
        elapsed = time.time() - start
        console.print(f"\n  [green]Indexing completed in {elapsed:.1f}s[/]")

    # Build RAG chain
    console.print("\nBuilding query chain...")
    
    # Show vector store stats
    try:
        collection = vectorstore._collection
        count = collection.count()
        console.print(f"[dim]Vector store contains {count} document chunks[/]")
        if count == 0:
            console.print("[yellow]⚠ Warning: Vector store is empty! Consider re-indexing.[/]")
    except Exception as e:
        console.print(f"[dim]Could not get vector store stats: {e}[/]")
    
    chain = build_chain(vectorstore, ollama_model, retrieval_k, directories)
    console.print("[green]Ready![/]\n")

    console.rule("[dim]Ask questions about your files. Type 'quit' to stop.[/]")
    console.print()

    # Interactive loop
    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        with console.status("[bold green]Searching and thinking...[/]"):
            try:
                response = chain.invoke(query)
            except Exception as e:
                console.print(f"\n[red]Error:[/] {e}\n")
                continue

        console.print()
        console.print(Panel(Markdown(response), title="Assistant", border_style="green"))
        console.print()


if __name__ == "__main__":
    main()
