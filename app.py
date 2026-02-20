"""
Streamlit web UI for the Offline File System Assistant.

Run with:
    streamlit run app.py
"""

import os
import shutil

import streamlit as st

from src.config import load_config, save_config, CHROMA_DIR
from src.indexing import (
    scan_directories,
    build_vectorstore,
    load_existing_vectorstore,
    index_directory,
    remove_directory_from_store,
)
from src.chain import build_chain
from visualize import build_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Offline File Assistant",
    page_icon="🗂",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources
# Prefixing a parameter with _ tells Streamlit not to hash it.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector store...")
def _get_vectorstore(embedding_model: str):
    return load_existing_vectorstore(embedding_model)


@st.cache_resource(show_spinner="Building agent...")
def _get_chain(_vectorstore, ollama_model: str, retrieval_k: int, directories: tuple):
    return build_chain(_vectorstore, ollama_model, retrieval_k, list(directories))


@st.cache_data(show_spinner="Rendering graph...")
def _get_graph_source(directories: tuple, max_depth: int, max_files: int) -> str:
    dot = build_graph(list(directories), max_depth=max_depth, max_files=max_files)
    return dot.source


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session(config: dict) -> None:
    """Seed session state from config on the very first run."""
    if "directories" not in st.session_state:
        st.session_state.directories = list(config.get("directories", []))


def _persist_directories(config: dict) -> None:
    """Write the current session directories back to config.yaml."""
    config["directories"] = list(st.session_state.directories)
    save_config(config)


# ---------------------------------------------------------------------------
# Sidebar — directory manager + global actions
# ---------------------------------------------------------------------------

def render_sidebar(config: dict) -> bool:
    """
    Render the sidebar.  Returns True when 'Re-index All' is clicked.

    Directory changes (add / remove) are handled here directly so the rest
    of main() only needs to react to the returned flag.
    """
    ollama_model: str = config.get("ollama_model", "qwen3:8b")
    embedding_model: str = config.get("embedding_model", "nomic-embed-text")
    chunk_size: int = config.get("chunk_size", 1000)
    chunk_overlap: int = config.get("chunk_overlap", 200)

    with st.sidebar:
        st.title("Offline File Assistant")
        st.caption(f"Model: `{ollama_model}` · Embed: `{embedding_model}`")
        st.divider()

        # ── Indexed directories ───────────────────────────────────────────
        st.markdown("**Indexed directories**")

        dirs = st.session_state.directories
        for i, d in enumerate(dirs):
            col_path, col_btn = st.columns([5, 1])
            col_path.code(d, language=None)
            if col_btn.button("✕", key=f"rm_{i}", help="Remove and un-index this directory"):
                vectorstore = _get_vectorstore(embedding_model)
                if vectorstore is not None:
                    with st.spinner(f"Removing {os.path.basename(d)} from index..."):
                        n = remove_directory_from_store(vectorstore, d)
                    st.toast(f"Removed {n} chunks for {os.path.basename(d)}")
                st.session_state.directories.pop(i)
                _persist_directories(config)
                st.cache_data.clear()   # refresh graph
                st.rerun()

        # ── Add new directory ─────────────────────────────────────────────
        st.markdown("**Add a directory**")
        new_path = st.text_input(
            "Directory path",
            placeholder=r"C:\Users\You\Documents",
            label_visibility="collapsed",
        )
        if st.button("Add & Index", use_container_width=True):
            new_path = new_path.strip()
            if not new_path:
                st.warning("Enter a path first.")
            elif not os.path.isdir(new_path):
                st.error(f"Path does not exist: {new_path}")
            elif new_path in st.session_state.directories:
                st.warning("That directory is already indexed.")
            else:
                vectorstore = _get_vectorstore(embedding_model)
                if vectorstore is None:
                    st.error("No index exists yet. Use 'Re-index All' first.")
                else:
                    with st.spinner(f"Indexing {os.path.basename(new_path)}..."):
                        n = index_directory(
                            vectorstore, new_path, ollama_model, chunk_size, chunk_overlap
                        )
                    st.session_state.directories.append(new_path)
                    _persist_directories(config)
                    st.cache_data.clear()   # refresh graph
                    st.toast(f"Added {n} chunks from {os.path.basename(new_path)}")
                    st.rerun()

        st.divider()

        # ── Global actions ────────────────────────────────────────────────
        return st.button(
            "Re-index All",
            use_container_width=True,
            type="primary",
            help="Wipe and rebuild the entire index from all configured directories.",
        )


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_chat(chain) -> None:
    st.subheader("Ask about your files")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Where did I save the AI documentation?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Graph tab
# ---------------------------------------------------------------------------

def render_graph(directories: list[str]) -> None:
    st.subheader("File System Graph")

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        max_depth = st.number_input("Max depth", min_value=1, max_value=10, value=3, step=1)
    with col2:
        max_files = st.number_input("Max files/dir", min_value=5, max_value=100, value=20, step=5)

    if not directories:
        st.info("No directories configured.")
        return

    dot_source = _get_graph_source(tuple(directories), int(max_depth), int(max_files))
    st.graphviz_chart(dot_source, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    embedding_model: str = config.get("embedding_model", "nomic-embed-text")
    ollama_model: str = config.get("ollama_model", "qwen3:8b")
    retrieval_k: int = config.get("retrieval_k", 5)
    chunk_size: int = config.get("chunk_size", 1000)
    chunk_overlap: int = config.get("chunk_overlap", 200)

    _init_session(config)

    reindex_all = render_sidebar(config)

    st.title("Offline File System Assistant")

    directories: list[str] = st.session_state.directories

    # ── Re-index all ─────────────────────────────────────────────────────────
    if reindex_all:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        st.cache_resource.clear()
        st.cache_data.clear()

        with st.spinner("Indexing files — this may take several minutes..."):
            documents = scan_directories(directories, ollama_model=ollama_model)
            build_vectorstore(documents, embedding_model, chunk_size, chunk_overlap)

        st.success("Indexing complete!")
        st.rerun()

    # ── Load vectorstore (warn if missing) ───────────────────────────────────
    vectorstore = _get_vectorstore(embedding_model)
    if vectorstore is None:
        st.warning(
            "No index found. Click **Re-index All** in the sidebar to scan "
            "your directories and build the vector store."
        )
        st.stop()

    chain = _get_chain(vectorstore, ollama_model, retrieval_k, tuple(directories))

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_chat, tab_graph = st.tabs(["Chat", "File System Graph"])

    with tab_chat:
        render_chat(chain)

    with tab_graph:
        render_graph(directories)


if __name__ == "__main__":
    main()
