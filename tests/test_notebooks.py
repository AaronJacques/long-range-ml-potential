"""Static checks of the notebooks under ``notebooks/`` (T16, B26).

The notebooks are committed without outputs and without machine-specific paths, and
they may only call ``lrmlp`` functions that exist. Nothing is executed: the import
lines of the code cells are parsed with :mod:`ast` and resolved with
:mod:`importlib`.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.rglob("*.ipynb"))
#: Absolute paths: Windows drives (``D:\``, ``C:/``), macOS/Linux home directories.
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _code(nb: dict) -> list[str]:
    return [_source(cell) for cell in nb["cells"] if cell["cell_type"] == "code"]


def _lrmlp_imports(code: str) -> list[tuple[str, str | None]]:
    """Return ``(module, name)`` of every ``lrmlp`` import (``name`` None for ``import``)."""
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "lrmlp":
                found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend(
                (alias.name, None) for alias in node.names if alias.name.split(".")[0] == "lrmlp"
            )
    return found


def test_notebooks_exist() -> None:
    assert NOTEBOOK_DIR / "training_curves.ipynb" in NOTEBOOKS


@pytest.fixture(params=NOTEBOOKS, ids=lambda p: p.relative_to(REPO).as_posix())
def notebook(request: pytest.FixtureRequest) -> tuple[Path, dict]:
    return request.param, _load(request.param)


def test_nbformat_4(notebook: tuple[Path, dict]) -> None:
    _, nb = notebook
    assert nb["nbformat"] == 4
    assert isinstance(nb["nbformat_minor"], int)
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"code", "markdown", "raw"}
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], str | list)


def test_no_outputs(notebook: tuple[Path, dict]) -> None:
    _, nb = notebook
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"cell {i} has outputs"
            assert cell["execution_count"] is None, f"cell {i} has an execution count"


def test_no_absolute_paths(notebook: tuple[Path, dict]) -> None:
    path, _ = notebook
    text = path.read_text(encoding="utf-8")
    assert "D:\\\\" not in text
    match = ABSOLUTE_PATH.search(text)
    if match is not None:
        context = text[max(match.start() - 40, 0) : match.end() + 40]
        pytest.fail(f"absolute path near {context!r}")


def test_code_cells_parse(notebook: tuple[Path, dict]) -> None:
    _, nb = notebook
    for code in _code(nb):
        ast.parse(code)


def test_lrmlp_imports_exist(notebook: tuple[Path, dict]) -> None:
    _, nb = notebook
    imports = [item for code in _code(nb) for item in _lrmlp_imports(code)]
    assert imports, "the notebook does not use lrmlp"
    for module_name, name in imports:
        module = importlib.import_module(module_name)
        if name is not None and name != "*":
            assert hasattr(module, name), f"{module_name} has no attribute {name}"


def test_training_curves_uses_analysis() -> None:
    nb = _load(NOTEBOOK_DIR / "training_curves.ipynb")
    imports = {item for code in _code(nb) for item in _lrmlp_imports(code)}
    assert ("lrmlp.analysis.plots", "plot_training_log") in imports
    assert ("lrmlp.analysis.weights", "plot_weight_distribution") in imports
    code = "\n".join(_code(nb))
    assert "RUN_DIRS" in code
    assert "matplotlib" not in code, "the figures are drawn by lrmlp.analysis"
    assert nb["cells"][0]["cell_type"] == "markdown"
    intro = _source(nb["cells"][0])
    for figure in ("5.1", "5.2", "A.1", "A.6", "5.3"):
        assert figure in intro


@pytest.mark.parametrize(
    ("text", "absolute"),
    [
        (r"D:\Uni\Master", True),
        ("C:/data", True),
        ("/Users/someone/runs", True),
        ("/home/someone/runs", True),
        ('Path("../runs")', False),
        ("Fig. A.1", False),
    ],
)
def test_absolute_path_pattern(text: str, absolute: bool) -> None:
    assert (ABSOLUTE_PATH.search(text) is not None) is absolute
