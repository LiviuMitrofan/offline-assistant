"""
Tools for the Offline File System Assistant agent.

Each tool is decorated with @tool so LangChain's AgentExecutor can
discover and invoke them automatically. The vectorstore and config
are injected via `build_tools()` to avoid global state.
"""

import os
from langchain_core.tools import tool
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Tool factory — call this once and pass the returned list to the agent
# ---------------------------------------------------------------------------

def build_tools(vectorstore: Chroma, indexed_directories: list[str]):
    """
    Return all agent tools with the vectorstore and config already bound.

    Args:
        vectorstore:        The Chroma instance built during indexing.
        indexed_directories: Root directories from config.yaml — used to
                             scope disk-usage calculations.
    """

    # ------------------------------------------------------------------
    # 1. Semantic file search
    #    Covers: "Where's the documentation for project X?",
    #            "Find a picture of a cat", "locate design diagrams…"
    # ------------------------------------------------------------------
    @tool
    def search_files(query: str, k: int = 8, extension: str = "") -> str:
        """
        Search the file index using natural language.

        Use this for questions like:
          - "Where is <filename>?"
          - "Find files related to <topic>"
          - "Locate design diagrams about microservices"
          - "Find a picture of a cat"

        Args:
            query:     Natural-language description of what to find.
            k:         Number of results to return (default 8).
            extension: Optional file extension filter, e.g. ".pdf", ".docx".
                       Leave empty to search all file types.

        Returns:
            Formatted list of matching files with paths.
        """
        where_filter = None
        if extension:
            ext = extension if extension.startswith(".") else f".{extension}"
            where_filter = {"extension": {"$eq": ext}}

        try:
            # Search with higher k to get more candidates for exact matching
            docs = vectorstore.similarity_search(
                query,
                k=min(k * 3, 50),  # Get more candidates for better exact matching
                filter=where_filter,
            )
        except Exception as e:
            return f"Search failed: {e}"

        if not docs:
            return "No matching files found."

        # Extract potential filename from query (look for patterns like "file.pdf")
        query_lower = query.lower()
        potential_filename = None
        
        # Simple filename extraction - look for common patterns
        # Match: "filename.pdf", "filename", etc.
        import re as _re
        filename_match = _re.search(r'([\w\-_]+\.\w+|[\w\-_]+)', query_lower)
        if filename_match:
            potential_filename = filename_match.group(1).lower()

        # Prioritize exact filename matches
        exact_matches = []
        partial_matches = []
        other_docs = []

        for doc in docs:
            path = doc.metadata.get("source", "unknown path")
            fname = doc.metadata.get("filename", os.path.basename(path))
            fname_lower = fname.lower()

            # Check for exact filename match
            if potential_filename:
                if fname_lower == potential_filename or fname_lower.endswith(potential_filename):
                    exact_matches.append((doc, path, fname))
                    continue
                elif potential_filename in fname_lower or fname_lower in potential_filename:
                    partial_matches.append((doc, path, fname))
                    continue

            other_docs.append((doc, path, fname))

        # Reorder: exact matches first, then partial, then others
        ordered_docs = exact_matches + partial_matches + other_docs

        # Limit to requested k results
        ordered_docs = ordered_docs[:k]

        if not ordered_docs:
            return "No matching files found."

        seen_paths: set[str] = set()
        lines: list[str] = []
        for doc, path, fname in ordered_docs:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            size = doc.metadata.get("size_bytes", "?")
            mod = doc.metadata.get("modified", "?")
            lines.append(f"• {fname}\n  Path: {path}\n  Size: {size} bytes  |  Modified: {mod}")

        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 2. Recent files
    #    Covers: "Help me find the last docx I worked on"
    # ------------------------------------------------------------------
    @tool
    def get_recent_files(extension: str = "", n: int = 10) -> str:
        """
        Return the N most recently modified files, optionally filtered by type.

        Use this for questions like:
          - "What was the last file I worked on?"
          - "Find the most recent Word document"
          - "Show me recently modified PDFs"

        Args:
            extension: Optional filter, e.g. ".docx", ".pdf".  Leave empty for all.
            n:         Number of files to return (default 10).

        Returns:
            Sorted list of recent files with paths and modification dates.
        """
        try:
            # Pull a broad sample from the vector store then re-sort by metadata
            all_docs = vectorstore.similarity_search("file", k=300)
        except Exception as e:
            return f"Could not retrieve files: {e}"

        seen: dict[str, dict] = {}
        for doc in all_docs:
            path = doc.metadata.get("source")
            if not path or path in seen:
                continue
            ext = doc.metadata.get("extension", "")
            if extension:
                target = extension if extension.startswith(".") else f".{extension}"
                if ext.lower() != target.lower():
                    continue
            seen[path] = doc.metadata

        # Sort by modification time descending
        sorted_files = sorted(
            seen.values(),
            key=lambda m: m.get("modified", ""),
            reverse=True,
        )[:n]

        if not sorted_files:
            kind = f"{extension} " if extension else ""
            return f"No {kind}files found in the index."

        lines = []
        for m in sorted_files:
            lines.append(
                f"• {m.get('filename', '?')}\n"
                f"  Path: {m.get('source', '?')}\n"
                f"  Modified: {m.get('modified', '?')}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 3. Folder summary / topic extraction
    #    Covers: "Summarize the <folder> folder",
    #            "Key topics across my ML research papers",
    #            "All notes about Kubernetes"
    # ------------------------------------------------------------------
    @tool
    def summarize_folder(folder_path: str, max_docs: int = 20) -> str:
        """
        Retrieve every indexed file inside a folder and return a combined
        summary of their names, types, and content snippets.

        Use this for:
          - "Summarize the contents of the Projects folder"
          - "What topics appear in my research papers?"
          - "Give me a summary of all notes about Kubernetes"

        The LLM calling this tool should then synthesize the returned raw
        context into a coherent answer for the user.

        Args:
            folder_path: Full or partial path of the folder to inspect.
            max_docs:    Maximum chunks to retrieve (default 20).

        Returns:
            Raw content blocks for each file found in that folder.
        """
        try:
            # Search broadly for anything living under that folder path
            docs = vectorstore.similarity_search(folder_path, k=max_docs)
        except Exception as e:
            return f"Could not search folder: {e}"

        # Keep only docs whose path contains the folder substring
        folder_norm = os.path.normpath(folder_path).lower()
        matching = [
            d for d in docs
            if folder_norm in os.path.normpath(d.metadata.get("source", "")).lower()
        ]

        # Fallback: if none matched by path filter, return all (user may have
        # used a partial/alias name and the LLM can still extract info)
        if not matching:
            matching = docs

        if not matching:
            return f"No files found for folder: {folder_path}"

        parts = []
        seen_paths: set[str] = set()
        for doc in matching:
            path = doc.metadata.get("source", "?")
            if path in seen_paths:
                continue
            seen_paths.add(path)
            snippet = doc.page_content[:600].replace("\n", " ")
            parts.append(
                f"FILE: {doc.metadata.get('filename', path)}\n"
                f"PATH: {path}\n"
                f"TYPE: {doc.metadata.get('extension', '?')}\n"
                f"SIZE: {doc.metadata.get('size_bytes', '?')} bytes\n"
                f"SNIPPET: {snippet}"
            )

        header = f"Found {len(seen_paths)} file(s) related to '{folder_path}':\n\n"
        return header + "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # 4. Disk usage
    #    Covers: "Which folders take up the most space on my laptop?"
    # ------------------------------------------------------------------
    @tool
    def get_folder_sizes(root: str = "", top_n: int = 10) -> str:
        """
        Calculate and rank folders by total disk usage.

        Use this for:
          - "Which folders take up the most space?"
          - "What's using disk space in my Documents folder?"

        Args:
            root:  Directory to analyse. Defaults to all configured directories.
            top_n: How many top folders to return (default 10).

        Returns:
            Ranked list of folders with their sizes in MB.
        """
        scan_roots = [root] if root else indexed_directories

        folder_sizes: dict[str, int] = {}

        for scan_root in scan_roots:
            scan_root = os.path.expanduser(scan_root)
            if not os.path.isdir(scan_root):
                continue
            for dirpath, _dirs, files in os.walk(scan_root):
                total = 0
                for fname in files:
                    fpath = os.path.join(dirpath, fname)
                    try:
                        total += os.path.getsize(fpath)
                    except OSError:
                        pass
                folder_sizes[dirpath] = folder_sizes.get(dirpath, 0) + total

        if not folder_sizes:
            return "Could not calculate folder sizes (no accessible directories)."

        ranked = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]

        lines = []
        for path, size_bytes in ranked:
            size_mb = size_bytes / (1024 * 1024)
            lines.append(f"• {size_mb:>8.1f} MB  —  {path}")

        return "Folders by size (largest first):\n\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # 5. Content-pattern search
    #    Covers: "Documents with client names or emails",
    #            "Files that appear to contain financial data"
    # ------------------------------------------------------------------
    @tool
    def search_by_content_pattern(description: str, k: int = 10) -> str:
        """
        Find files whose *content* matches a semantic description or data pattern.

        Use this for questions like:
          - "Show me documents that include email addresses"
          - "List files that appear to contain financial data"
          - "Find files mentioning invoices or contracts"

        This is different from `search_files` because it focuses on the
        *body text* of indexed documents, not just their names/paths.

        Args:
            description: What kind of content to look for in plain English.
            k:           Number of results (default 10).

        Returns:
            Matching files with a short content snippet as evidence.
        """
        try:
            docs = vectorstore.similarity_search(description, k=k)
        except Exception as e:
            return f"Search failed: {e}"

        if not docs:
            return "No matching documents found."

        # Prefer chunks that have actual extracted content
        content_docs = [d for d in docs if "Content:" in d.page_content]
        if not content_docs:
            content_docs = docs

        seen: set[str] = set()
        lines = []
        for doc in content_docs:
            path = doc.metadata.get("source", "?")
            if path in seen:
                continue
            seen.add(path)

            # Extract the content portion of the chunk
            content_part = doc.page_content
            if "Content:" in content_part:
                content_part = content_part.split("Content:", 1)[1].strip()

            snippet = content_part[:300].replace("\n", " ")
            lines.append(
                f"• {doc.metadata.get('filename', path)}\n"
                f"  Path: {path}\n"
                f"  Evidence: …{snippet}…"
            )

        return f"Files matching '{description}':\n\n" + "\n\n".join(lines)

    # ------------------------------------------------------------------
    # 6. Get full file details + all indexed content chunks
    #    Covers: "What is the content of file X about?",
    #            "Tell me more about this specific file"
    # ------------------------------------------------------------------
    @tool
    def get_file_details(filename_or_path: str) -> str:
        """
        Return all indexed content (metadata + every text chunk) for a
        specific file, identified by name or partial/full path.

        Use this when the user asks what a file is about, wants a summary
        of a specific file, or needs in-depth info about one particular file.

        Args:
            filename_or_path: The file name (e.g. "report.pdf") or any part
                              of its path.

        Returns:
            Full metadata and all indexed content for the matched file.
        """
        # Step 1: Semantic search to resolve the real stored path
        try:
            candidates = vectorstore.similarity_search(filename_or_path, k=15)
        except Exception as e:
            return f"Lookup failed: {e}"

        name_lower = os.path.basename(filename_or_path).lower()

        # Prefer exact filename match, then partial, then first result
        resolved_path: str | None = None
        for doc in candidates:
            fname = doc.metadata.get("filename", "").lower()
            if fname == name_lower:
                resolved_path = doc.metadata.get("source")
                break
        if not resolved_path:
            for doc in candidates:
                fname = doc.metadata.get("filename", "").lower()
                if name_lower in fname or fname in name_lower:
                    resolved_path = doc.metadata.get("source")
                    break
        if not resolved_path and candidates:
            resolved_path = candidates[0].metadata.get("source")

        if not resolved_path:
            return f"File '{filename_or_path}' not found in the index."

        # Step 2: Fetch ALL chunks for this file via metadata filter so we
        #         get the full content, not just the closest-embedding chunk.
        try:
            result = vectorstore._collection.get(
                where={"source": {"$eq": resolved_path}},
                include=["documents", "metadatas"],
            )
            all_chunks: list[str] = result.get("documents") or []
            all_meta: list[dict] = result.get("metadatas") or []
        except Exception:
            # Graceful fallback to the candidate docs already retrieved
            all_chunks = [d.page_content for d in candidates if d.metadata.get("source") == resolved_path]
            all_meta = [d.metadata for d in candidates if d.metadata.get("source") == resolved_path]

        if not all_chunks:
            return f"File '{filename_or_path}' not found in the index."

        meta = all_meta[0] if all_meta else {}

        # Build header from metadata
        header = (
            f"FILENAME: {meta.get('filename', '?')}\n"
            f"PATH:     {meta.get('source', '?')}\n"
            f"TYPE:     {meta.get('extension', '?')}\n"
            f"SIZE:     {meta.get('size_bytes', '?')} bytes\n"
            f"MODIFIED: {meta.get('modified', '?')}\n"
        )

        # Extract content sections from chunks (skip pure-metadata chunks)
        content_parts: list[str] = []
        for chunk in all_chunks:
            if "Content:" in chunk:
                part = chunk.split("Content:", 1)[1].strip()
                if part:
                    content_parts.append(part)

        if not content_parts:
            return header + "\n(No text content indexed for this file — binary or unsupported format)"

        combined = "\n\n".join(content_parts)
        # Cap at 6000 chars to keep context manageable
        if len(combined) > 6000:
            combined = combined[:6000] + "\n… [content truncated — file has more text]"

        return header + f"\nCONTENT:\n{combined}"

    # Return all tools as a list
    return [
        search_files,
        get_recent_files,
        summarize_folder,
        get_folder_sizes,
        search_by_content_pattern,
        get_file_details,
    ]