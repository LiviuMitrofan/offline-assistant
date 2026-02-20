# Offline File System Assistant

A fully offline, locally-running AI assistant that indexes your file system and answers natural language questions about your files — their names, locations, metadata, and content.

Built with **LangChain**, **Ollama**, **ChromaDB**, **Streamlit**, and **Graphviz**.

---

## Features

- **Natural language file search** — find files by description, not just name
- **Content-aware answers** — reads PDFs, DOCX, images (via vision LLM), and plain text
- **LLM-generated summaries & keywords** — enriches every indexed file for smarter retrieval
- **Directory overview documents** — instant answers to "what's in this folder?"
- **Conversation memory** — the agent remembers context across turns in a session
- **Graphical file-system view** — interactive tree diagram of your directories
- **Two interfaces** — a Streamlit web UI and a classic CLI

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [Ollama](https://ollama.com) | Must be running locally |
| [Graphviz](https://graphviz.org/download/) | Required for the graph view (system binary) |

### Ollama models needed

```bash
ollama pull qwen3:8b           # chat / reasoning / summaries
ollama pull nomic-embed-text   # embeddings
```

> The image OCR feature uses `glm-ocr` if available (`ollama pull glm-ocr`), otherwise images are indexed by metadata only.

---

## Installation

```bash
# Clone / unzip the project, then:
uv sync          # installs all Python dependencies

# Or with pip:
pip install -e .
```

> **Graphviz system binary** — the Python `graphviz` package is a wrapper that needs the `dot` executable.
> Download and install from [graphviz.org/download](https://graphviz.org/download/).
> On Windows, add `C:\Program Files\Graphviz\bin` to your system PATH (the app auto-detects this location as a fallback).

---

## Configuration

Edit `config.yaml` before the first run:

```yaml
# Directories to scan and index
directories:
  - "C:/Users/YourName/Documents"
  - "C:/Users/YourName/Desktop/projects"

# Text splitting
chunk_size: 1000
chunk_overlap: 200

# Ollama models
ollama_model: "qwen3:8b"
embedding_model: "nomic-embed-text"

# Retrieval
retrieval_k: 5
```

### Configuration reference

| Key | Default | Description |
|---|---|---|
| `directories` | — | Paths to scan. Supports `~` expansion. |
| `chunk_size` | `1000` | Characters per text chunk |
| `chunk_overlap` | `200` | Overlap between adjacent chunks |
| `ollama_model` | `qwen3:8b` | Model used for chat, summaries, and keywords |
| `embedding_model` | `nomic-embed-text` | Model used for vector embeddings |
| `retrieval_k` | `5` | Number of chunks retrieved per query |

---

## Running the app

### Web UI (Streamlit) — recommended

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Two tabs are available:

- **Chat** — conversational assistant with full memory across turns
- **File System Graph** — interactive graphviz tree of your directories

On first launch, click **Re-index Files** in the sidebar to build the index.

### CLI

```bash
python main.py
```

Interactive terminal session. On first run you are asked whether to build a fresh index or load the existing one.

### File system visualizer (standalone)

```bash
# SVG output, opens automatically in browser
python visualize.py

# PNG output
python visualize.py --format png

# Limit tree depth to 2 levels
python visualize.py --depth 2

# Cap files shown per folder
python visualize.py --max-files 15

# Render without opening
python visualize.py --no-view

# Custom output path (no extension)
python visualize.py --output C:/Users/YourName/Desktop/my_fs
```

---

## Example questions

```
Where did I save the documentation for project X?
Help me find the last docx I worked on.
Find a picture of a cat.
Summarize the contents of the projects folder.
What are the key topics across my research papers on machine learning?
Give me a summary of all notes I've written about Kubernetes.
Which folders take up the most space on my laptop?
Show me documents that include client names or email addresses.
List files that appear to contain financial data.
Locate all design diagrams involving microservices architectures.
What is the content of report.pdf about?
Does that file contain a section on security?
```

---

## Supported file types

| Category | Extensions |
|---|---|
| Plain text / Code | `.txt`, `.md`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.rb`, `.php`, `.html`, `.css`, `.json`, `.yaml`, `.xml`, `.csv`, `.sql`, `.sh`, `.bat`, `.toml`, `.ini`, `.log`, and more |
| PDF | `.pdf` |
| Word | `.docx` |
| Images | `.png`, `.jpg`, `.jpeg` — described via vision LLM (`glm-ocr`) |
| Other | All other files are indexed by metadata (name, path, size, dates) and remain searchable |

---

## Indexing pipeline

On each (re-)index run, every file produces up to three vector store documents:

| Document type | Purpose |
|---|---|
| **Content chunk(s)** | Full extracted text split into overlapping chunks — used for precise retrieval |
| **Summary document** | One short LLM-written summary per file, heavily searchable for broad topic queries |
| **Directory overview** | One document per folder listing all files with names, sizes, dates, and summary snippets |

Keywords and a summary are also stored as metadata on every content chunk, making them available as filters.

---

## Architecture

```
config.yaml
    │
    ├── src/config.py       — loads config.yaml
    ├── src/file_utils.py   — file reading (text, PDF, DOCX, images) + LLM enrichment
    ├── src/indexing.py     — directory scan, document building, Chroma vector store
    ├── src/tools.py        — LangChain agent tools (search, details, recent files, …)
    ├── src/chain.py        — LangChain agent + conversation history adapter
    │
    ├── main.py             — CLI entry point
    ├── app.py              — Streamlit web UI entry point
    └── visualize.py        — Graphviz file system tree renderer
```

### Data flow

```
Directories → file_utils (extract text) → indexing (chunk + embed) → ChromaDB
                                                ↓ LLM calls
                                        summaries + keywords stored per file

User query → agent (chain.py) → tools (search ChromaDB) → synthesised answer
```

Everything runs 100% offline. No data leaves your machine.
