"""
File system visualization using Graphviz.

Reads the directories from config.yaml and renders a tree diagram
showing every subdirectory (as a cluster) and every file (as a node),
coloured and shaped by file type.

Usage:
    uv run python visualize.py                   # SVG, opens automatically
    uv run python visualize.py --format png      # PNG instead
    uv run python visualize.py --depth 2         # limit tree depth
    uv run python visualize.py --max-files 20    # cap files shown per folder
    uv run python visualize.py --no-view         # render but don't open
"""

import argparse
import os
import re
import sys
from pathlib import Path

import graphviz
import yaml

# ---------------------------------------------------------------------------
# Windows: Graphviz is often installed but its bin dir is not on PATH.
# Check the two standard install locations and patch os.environ["PATH"] so
# the graphviz Python package can find dot.exe for this process.
# ---------------------------------------------------------------------------
_GRAPHVIZ_CANDIDATES = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin",
]
for _gv_bin in _GRAPHVIZ_CANDIDATES:
    if Path(_gv_bin).is_dir() and _gv_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _gv_bin + os.pathsep + os.environ.get("PATH", "")
        break

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
OUTPUT_PATH = Path(__file__).parent / "fs_visualization"

# ---------------------------------------------------------------------------
# File-type palette
# (fillcolor, shape, text label prefix)
# ---------------------------------------------------------------------------
EXT_STYLES: dict[str, dict] = {
    # Documents
    ".pdf":  {"fillcolor": "#ffb3b3", "shape": "note",      "icon": "PDF"},
    ".docx": {"fillcolor": "#b3d4ff", "shape": "note",      "icon": "DOC"},
    ".doc":  {"fillcolor": "#b3d4ff", "shape": "note",      "icon": "DOC"},
    ".xlsx": {"fillcolor": "#b3ffcc", "shape": "note",      "icon": "XLS"},
    ".xls":  {"fillcolor": "#b3ffcc", "shape": "note",      "icon": "XLS"},
    ".pptx": {"fillcolor": "#ffd9b3", "shape": "note",      "icon": "PPT"},
    # Text / Markup
    ".txt":  {"fillcolor": "#d4edda", "shape": "note",      "icon": "TXT"},
    ".md":   {"fillcolor": "#d4edda", "shape": "note",      "icon": "MD "},
    ".rst":  {"fillcolor": "#d4edda", "shape": "note",      "icon": "RST"},
    # Images
    ".png":  {"fillcolor": "#fff3b3", "shape": "box",       "icon": "IMG"},
    ".jpg":  {"fillcolor": "#fff3b3", "shape": "box",       "icon": "IMG"},
    ".jpeg": {"fillcolor": "#fff3b3", "shape": "box",       "icon": "IMG"},
    ".gif":  {"fillcolor": "#fff3b3", "shape": "box",       "icon": "IMG"},
    ".webp": {"fillcolor": "#fff3b3", "shape": "box",       "icon": "IMG"},
    ".svg":  {"fillcolor": "#fff3b3", "shape": "box",       "icon": "SVG"},
    # Code
    ".py":   {"fillcolor": "#e2d9f3", "shape": "component", "icon": "PY "},
    ".js":   {"fillcolor": "#e2d9f3", "shape": "component", "icon": "JS "},
    ".ts":   {"fillcolor": "#e2d9f3", "shape": "component", "icon": "TS "},
    ".java": {"fillcolor": "#e2d9f3", "shape": "component", "icon": "JAV"},
    ".go":   {"fillcolor": "#e2d9f3", "shape": "component", "icon": "GO "},
    ".rs":   {"fillcolor": "#e2d9f3", "shape": "component", "icon": "RS "},
    ".c":    {"fillcolor": "#e2d9f3", "shape": "component", "icon": "C  "},
    ".cpp":  {"fillcolor": "#e2d9f3", "shape": "component", "icon": "C++"},
    # Data / Config
    ".json": {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "JSON"},
    ".yaml": {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "YAML"},
    ".yml":  {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "YAML"},
    ".csv":  {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "CSV"},
    ".xml":  {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "XML"},
    ".sql":  {"fillcolor": "#d9f3e2", "shape": "box",       "icon": "SQL"},
}

DEFAULT_STYLE = {"fillcolor": "#e9ecef", "shape": "box", "icon": "   "}

# Alternating cluster fill colours for depth levels (creates visual hierarchy)
CLUSTER_FILLS = [
    "#dce8f8",   # depth 0 — strong blue tint
    "#eef4fb",   # depth 1
    "#f5f9fd",   # depth 2+
]
CLUSTER_BORDER = "#4a90d9"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_id(path: str) -> str:
    """Turn an arbitrary path into a valid Graphviz node/subgraph identifier."""
    return "n_" + re.sub(r"[^a-zA-Z0-9]", "_", path)


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.1f} KB"
    return f"{size_bytes} B"


def _file_style(filename: str) -> dict:
    ext = Path(filename).suffix.lower()
    return EXT_STYLES.get(ext, DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# Recursive graph builder
# ---------------------------------------------------------------------------

def _add_directory(
    parent_graph: graphviz.Digraph,
    dir_path: str,
    depth: int,
    max_depth: int | None,
    max_files: int,
) -> None:
    dir_name = os.path.basename(dir_path) or dir_path
    cluster_name = "cluster_" + _node_id(dir_path)
    fill = CLUSTER_FILLS[min(depth, len(CLUSTER_FILLS) - 1)]

    try:
        entries = sorted(os.scandir(dir_path), key=lambda e: (e.is_file(), e.name.lower()))
    except PermissionError:
        return

    subdirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
    files   = [e for e in entries if e.is_file() and not e.name.startswith(".")]

    with parent_graph.subgraph(name=cluster_name) as sub:
        # Cluster (directory) visual attrs
        sub.attr(
            label=f"[DIR]  {dir_name}",
            style="rounded,filled",
            fillcolor=fill,
            color=CLUSTER_BORDER,
            penwidth="1.5",
            fontname="Helvetica-Bold",
            fontsize="11",
            margin="10",
        )

        # File nodes
        shown = files[:max_files]
        for entry in shown:
            nid = _node_id(entry.path)
            style = _file_style(entry.name)
            try:
                size_label = _fmt_size(entry.stat().st_size)
            except OSError:
                size_label = "?"

            label = f"{entry.name}\n{size_label}"
            sub.node(
                nid,
                label=label,
                shape=style["shape"],
                style="filled",
                fillcolor=style["fillcolor"],
                fontname="Helvetica",
                fontsize="9",
                margin="0.05,0.03",
                height="0.4",
            )

        # Overflow indicator
        if len(files) > max_files:
            oid = _node_id(dir_path + "__overflow")
            sub.node(
                oid,
                label=f"… {len(files) - max_files} more file(s)",
                shape="plaintext",
                fontname="Helvetica-Oblique",
                fontsize="9",
                fontcolor="#888888",
            )

        # Recurse into subdirectories
        if max_depth is None or depth < max_depth:
            for entry in subdirs:
                _add_directory(sub, entry.path, depth + 1, max_depth, max_files)
        elif subdirs:
            # Show a collapsed indicator for hidden subdirectories
            cid = _node_id(dir_path + "__collapsed")
            sub.node(
                cid,
                label=f"[{len(subdirs)} subdirectory(ies) — depth limit reached]",
                shape="plaintext",
                fontname="Helvetica-Oblique",
                fontsize="9",
                fontcolor="#888888",
            )


def _add_legend(dot: graphviz.Digraph) -> None:
    """Add a compact legend cluster to the graph."""
    with dot.subgraph(name="cluster_legend") as leg:
        leg.attr(
            label="Legend",
            style="rounded,filled",
            fillcolor="#fafafa",
            color="#aaaaaa",
            fontname="Helvetica-Bold",
            fontsize="10",
            margin="10",
            rank="sink",
        )
        entries = [
            ("leg_pdf",   "PDF",      "#ffb3b3", "note"),
            ("leg_doc",   "DOCX",     "#b3d4ff", "note"),
            ("leg_img",   "Image",    "#fff3b3", "box"),
            ("leg_txt",   "Text/MD",  "#d4edda", "note"),
            ("leg_code",  "Code",     "#e2d9f3", "component"),
            ("leg_data",  "Data/Cfg", "#d9f3e2", "box"),
            ("leg_other", "Other",    "#e9ecef", "box"),
        ]
        # Chain them invisibly so they appear in a row
        prev = None
        for nid, label, fill, shape in entries:
            leg.node(
                nid,
                label=label,
                shape=shape,
                style="filled",
                fillcolor=fill,
                fontname="Helvetica",
                fontsize="9",
                height="0.35",
                width="0.8",
            )
            if prev:
                leg.edge(prev, nid, style="invis")
            prev = nid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    directories: list[str],
    max_depth: int | None = None,
    max_files: int = 30,
) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        name="FileSystem",
        comment="Offline Assistant — File System Visualization",
    )
    dot.attr(
        rankdir="LR",          # left-to-right layout for wide trees
        bgcolor="#ffffff",
        fontname="Helvetica",
        splines="ortho",       # right-angle edges — cleaner for tree layouts
        nodesep="0.3",
        ranksep="0.8",
        compound="true",       # allows edges between clusters
    )
    dot.attr("node", fontname="Helvetica", fontsize="10", penwidth="1.0")
    dot.attr("edge", color="#777777", arrowsize="0.5")

    for directory in directories:
        directory = os.path.expanduser(directory)
        if os.path.isdir(directory):
            _add_directory(dot, directory, depth=0, max_depth=max_depth, max_files=max_files)
        else:
            print(f"  Warning: '{directory}' does not exist, skipping.", file=sys.stderr)

    _add_legend(dot)
    return dot


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the configured file-system directories with Graphviz."
    )
    parser.add_argument(
        "--format", default="svg", choices=["svg", "png", "pdf"],
        help="Output format (default: svg)",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Max directory depth to render (default: unlimited)",
    )
    parser.add_argument(
        "--max-files", type=int, default=30,
        help="Max files shown per directory before truncation (default: 30)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Output file path without extension (default: fs_visualization)",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="Render but do not open the output file automatically",
    )
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        print(f"Error: config.yaml not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    directories: list[str] = config.get("directories", [])
    if not directories:
        print("Error: no directories configured in config.yaml", file=sys.stderr)
        sys.exit(1)

    plural = "y" if len(directories) == 1 else "ies"
    print(f"Building graph for {len(directories)} root director{plural}...")
    dot = build_graph(directories, max_depth=args.depth, max_files=args.max_files)

    out = dot.render(
        filename=args.output,
        format=args.format,
        view=not args.no_view,
        cleanup=True,          # remove the intermediate .gv source file
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
