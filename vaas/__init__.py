"""
Local namespace package shim so `python -m vaas.*` works without PYTHONPATH hacks.

When the project is installed (pip install -e .), the real package lives under
`src/vaas`. This module simply extends the package search path to include the
source tree so developers can run `python -m vaas.run_pipeline_v2` directly
from the repo root without manipulating sys.path.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_SRC_PATH = Path(__file__).resolve().parent.parent / "src" / "vaas"
if _SRC_PATH.exists():
    _src_str = str(_SRC_PATH)
    if _src_str not in __path__:
        __path__.append(_src_str)
