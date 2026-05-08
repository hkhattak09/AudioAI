"""Configuration utilities."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, data: dict) -> None:
    """Save a dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
