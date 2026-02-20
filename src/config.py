"""Application configuration loading."""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        print(f"Error: {CONFIG_PATH} not found. Please create it first.")
        sys.exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict) -> None:
    """Persist a config dict back to config.yaml."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
