"""File reading and metadata extraction utilities."""

import datetime
import re
from pathlib import Path


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Extensions we can read as plain text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".html",
    ".htm", ".json", ".yaml", ".yml", ".xml", ".csv", ".log", ".ini",
    ".cfg", ".toml", ".sh", ".bat", ".ps1", ".sql", ".r", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".lua", ".pl", ".env", ".gitignore",
    ".dockerfile", ".makefile",
}


def read_text_file(path: str) -> str | None:
    """Try to read a file as UTF-8 text."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def read_pdf(path: str) -> str | None:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


def read_docx(path: str) -> str | None:
    """Extract text from a .docx file."""
    try:
        import docx2txt

        text = docx2txt.process(path)
        return text if text and text.strip() else None
    except Exception:
        return None


def read_image(path: str) -> str | None:
    """Extract text from a image file."""
    try:
        from ollama import chat
        response = chat(
            model='glm-ocr',
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that explains images in a few sentences",
                },
                {
                    "role": "user",
                    "content": "Explain the image in a few sentences",
                    "images": [path],
                }
            ]
        )
        image_description = response.message.content
        return image_description
    except Exception:
        return None

def get_file_metadata(filepath: str) -> dict:
    """Extract metadata for a file."""
    p = Path(filepath)
    stat = p.stat()
    return {
        "source": str(p.resolve()),
        "filename": p.name,
        "extension": p.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "directory": str(p.parent.resolve()),
    }


def extract_content(filepath: str) -> str | None:
    """Extract text content from a file based on its extension."""
    ext = Path(filepath).suffix.lower()

    if ext in TEXT_EXTENSIONS or ext == "":
        return read_text_file(filepath)
    elif ext == ".pdf":
        return read_pdf(filepath)
    elif ext == ".docx":
        return read_docx(filepath)
    elif ext == ".png" or ext == ".jpg" or ext == ".jpeg":
        return read_image(filepath)
    else:
        # Unsupported binary file — content will be None
        return None


# ---------------------------------------------------------------------------
# LLM-assisted enrichment helpers
# ---------------------------------------------------------------------------

def generate_document_summary(content: str, model: str) -> str | None:
    """
    Ask Ollama to write a 2-3 sentence summary of the document content.

    Returns None on failure so callers can skip gracefully.
    Only the first 3 000 characters are sent to keep latency low.
    """
    if not content or not content.strip():
        return None
    snippet = content[:3000]
    try:
        from ollama import chat
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document summarizer. "
                        "Write only the summary — no preamble, no labels, no explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Summarize this document in 2-3 sentences:\n\n{snippet}",
                },
            ],
            options={"think": False, "temperature": 0, "num_predict": 200},
        )
        return _strip_think(response.message.content).strip() or None
    except Exception:
        return None


def extract_document_keywords(content: str, model: str) -> list[str]:
    """
    Ask Ollama to extract 5-8 keywords from the document content.

    Returns an empty list on failure.
    Only the first 2 000 characters are sent to keep latency low.
    """
    if not content or not content.strip():
        return []
    snippet = content[:2000]
    try:
        from ollama import chat
        response = chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract keywords. "
                        "Reply with ONLY a comma-separated list of keywords. "
                        "No other text, no numbering, no explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract 5-8 keywords from this:\n\n{snippet}",
                },
            ],
            options={"think": False, "temperature": 0, "num_predict": 100},
        )
        raw = _strip_think(response.message.content).strip()
        return [kw.strip() for kw in raw.split(",") if kw.strip()]
    except Exception:
        return []
