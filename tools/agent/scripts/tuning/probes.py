"""Probe loading and result persistence utilities."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def load_probes(path: Path | str) -> list[dict]:
    """Load probes from a YAML file.

    Supports two layouts:
      - Flat list: [{id, query, expected_decision, tags?}, ...]
      - Grouped:   {decisions: [{id, expected_decision, probes: [...]}]}
        where each inner probe inherits expected_decision from its group.
    """
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, list):
        return raw

    probes: list[dict] = []
    for group in raw.get("decisions", []):
        expected = group.get("expected_decision", group.get("id", ""))
        group_tags = group.get("tags", [])
        for p in group.get("probes", group.get("variants", [])):
            p.setdefault("expected_decision", expected)
            existing_tags = p.get("tags", [])
            p["tags"] = list(set(existing_tags + group_tags))
            probes.append(p)
    return probes if probes else raw.get("probes", [])


def save_results(
    output: dict[str, Any],
    filename: str,
    output_dir: Path | str | None = None,
) -> Path:
    """Write evaluation output to a JSON file, returning the path."""
    if output_dir is None:
        output_dir = Path.cwd() / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filename
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    return out_path
