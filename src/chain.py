"""RAG agent construction — replaces the simple chain with a tool-calling agent."""

import re
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent

from .tools import build_tools

# ---------------------------------------------------------------------------
# System prompt — plain instructions, NO ReAct text format.
# create_agent wires tools via the model's native function-calling API, so
# injecting the old Action/Observation template would cause the model to
# hallucinate fake tool traces instead of actually invoking tools.
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """You are an offline file system assistant. \
You have access to tools that let you search, analyse, and retrieve information \
about the user's locally indexed files.

RULES:
1. You MUST call a tool before answering — never answer from memory or make up file paths.
2. Pick the most specific tool for the question:
   - Finding a file by name/description → search_files
   - Most recently modified file        → get_recent_files
   - Summarising a folder or topic      → summarize_folder
   - Disk space / folder sizes          → get_folder_sizes
   - Files whose content matches a pattern → search_by_content_pattern
   - In-depth info about one specific file → get_file_details
3. Synthesise the tool results into a clear, direct answer.
4. Always include the full file path in your answer when relevant.
5. If no results are found, say so clearly — never invent paths or filenames.
"""


# ---------------------------------------------------------------------------
# Output cleaner — strips qwen3 <think> blocks if present
# ---------------------------------------------------------------------------
def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_chain(vectorstore: Chroma, ollama_model: str, retrieval_k: int, indexed_directories: list[str]):
    """
    Build and return an AgentExecutor wired up with all file-system tools.

    Args:
        vectorstore:          Chroma instance from indexing.
        ollama_model:         Ollama model name (e.g. "qwen3:8b").
        retrieval_k:          Default k passed to search tools (informational,
                              individual tools have their own defaults).
        indexed_directories:  Root dirs from config — needed by get_folder_sizes.

    Returns:
        An object with an `.invoke({"input": query})` interface, matching the
        original chain API so main.py needs minimal changes.
    """
    tools = build_tools(vectorstore, indexed_directories)

    # Lower temperature → more deterministic tool selection and path reporting
    # think=False → suppress qwen3 chain-of-thought tokens in the output stream
    llm = ChatOllama(
        model=ollama_model,
        temperature=0,
        num_ctx=8192,          # larger context for multi-tool conversations
        think=False,           # qwen3-specific: disable <think> blocks
    )

    # create_agent (langchain v1) uses the model's native tool-calling API.
    # system_prompt is injected as the system message; no ReAct format needed.
    agent = create_agent(
        model=llm,
        name="file_system_assistant",
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # ---------------------------------------------------------------------------
    # Thin wrapper so main.py can still call chain.invoke(query).
    # History is accumulated as the actual LangChain message objects returned
    # by the agent — this preserves tool_call_ids so ToolMessages stay linked
    # to their parent AIMessage, which is required for multi-turn correctness.
    # ---------------------------------------------------------------------------
    class _ChainAdapter:
        def __init__(self) -> None:
            # Stores the full list of LangChain message objects from all turns.
            self._history: list = []

        def invoke(self, query: str) -> str:
            messages = self._history + [{"role": "user", "content": query}]
            result = agent.invoke({"messages": messages})
            # result["messages"] is the complete thread including new AI/tool turns.
            self._history = list(result["messages"])
            return self._history[-1].content

        def reset(self) -> None:
            """Clear conversation history between independent sessions."""
            self._history.clear()

    return _ChainAdapter()