#!/usr/bin/env python3
"""
fix_notebooks.py
================
Repairs .ipynb files whose cell `source` lists were written without trailing
newlines.

The .ipynb spec stores a cell's text as a list of strings that Jupyter
reassembles with ``"".join(source)`` - so every entry except the last must end
with "\\n". A generator that used ``text.split("\\n")`` produces entries with no
newline at all, and every cell collapses onto a single line: valid JSON, valid
notebook structure, completely broken code.

Run it on a file or a directory. Idempotent: cells that are already correct are
left untouched.

    python fix_notebooks.py notebooks/
    python fix_notebooks.py notebooks/05_finetune.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def needs_fix(source: list) -> bool:
    """True when more than one entry exists and none of them ends with \\n."""
    return len(source) > 1 and not any(s.endswith("\n") for s in source)


def fix_source(source: list) -> list:
    """Rebuild the list so each line keeps its newline, except the last."""
    text = "\n".join(source)
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def fix_notebook(path: Path) -> tuple[int, int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    fixed = 0
    for cell in nb.get("cells", []):
        source = cell.get("source")
        if isinstance(source, str):
            cell["source"] = fix_source(source.split("\n"))
            fixed += 1
        elif isinstance(source, list) and needs_fix(source):
            cell["source"] = fix_source(source)
            fixed += 1
    if fixed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return fixed, len(nb.get("cells", []))


def verify(path: Path) -> list:
    """
    Validate the way Jupyter actually reads a notebook: join with "" and parse.
    Joining with "\\n" here would hide exactly the bug this script fixes.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell["source"])
        try:
            # compile(), not ast.parse(). Some errors - notably a walrus operator
            # in a comprehension's iterable - are raised when the symbol table is
            # built, which ast.parse() never does. ast.parse accepts them happily.
            compile(code, f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:
            problems.append(f"{path.name} cell {i}: {exc.msg} (line {exc.lineno})")
    return problems


def main(argv: list) -> int:
    targets = [Path(a) for a in argv[1:]] or [Path("notebooks")]
    files: list = []
    for t in targets:
        files.extend(sorted(t.glob("*.ipynb")) if t.is_dir() else [t])

    print(f"repairing {len(files)} notebook(s)\n")
    all_problems = []
    for path in files:
        fixed, total = fix_notebook(path)
        problems = verify(path)
        all_problems += problems
        status = "already correct" if not fixed else f"repaired {fixed}/{total} cells"
        mark = "ok " if not problems else "FAIL"
        print(f"  {mark} {path.name:<36} {status}")
        for p in problems:
            print(f"        {p}")

    print()
    if all_problems:
        print(f"{len(all_problems)} cell(s) still do not parse")
        return 1
    print("every code cell parses when joined the way Jupyter joins it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
