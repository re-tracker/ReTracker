from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


def test_offline_engine_has_no_duplicate_method_definitions() -> None:
    """Duplicate method defs are silent bugs in Python.

    If a method is defined twice in a class body, the last definition wins and
    the earlier one becomes dead code (easy to miss during refactors).
    """

    path = Path("retracker/inference/engines/offline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    engine_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ReTrackerEngine":
            engine_class = node
            break

    assert engine_class is not None, "Could not find ReTrackerEngine in offline engine module"

    method_names = [n.name for n in engine_class.body if isinstance(n, ast.FunctionDef)]
    counts = Counter(method_names)
    dupes = sorted(name for name, c in counts.items() if c > 1)
    assert dupes == [], f"Duplicate method definitions found in ReTrackerEngine: {dupes}"

