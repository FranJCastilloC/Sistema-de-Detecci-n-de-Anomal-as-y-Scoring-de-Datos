"""Make `src/` importable when running these scripts directly.

The project is normally installed with `uv pip install -e .`, which drops a
`.pth` file into site-packages. On macOS that file can end up carrying the BSD
`UF_HIDDEN` flag, and CPython's `site.addpackage` skips hidden `.pth` files
*silently* - the install looks present in `pip list` and the import still fails.

Rather than depend on that, every script puts `src/` on the path itself. The
editable install remains useful (it provides the console entry point) but is no
longer required for anything here to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def ensure_src_on_path() -> None:
    """Prepend the project's src directory to sys.path, once."""
    path = str(SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
