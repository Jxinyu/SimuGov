from pathlib import Path

from config.config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
settings = load_settings(PROJECT_ROOT / "config" / "config.yaml")
