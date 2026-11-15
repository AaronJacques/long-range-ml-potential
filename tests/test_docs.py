"""Checks of the user documentation: README.md, docs/*.md and CITATION.cff.

* every field of every configuration dataclass is documented in
  ``docs/configuration.md``, with the default of the code and the preset values of
  ``configs/``;
* all relative links (and their anchors) in README.md and docs/*.md resolve;
* every ``lrmlp <command>`` mentioned exists, and every ``lrmlp`` command line in a
  shell block parses and refers to a valid configuration;
* CITATION.cff has the required keys;
* the README quickstart and the Python example of docs/architecture.md run
  verbatim (marker ``slow``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lrmlp import config as cfg
from lrmlp.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
DOCS = REPO_ROOT / "docs"
CONFIGURATION = DOCS / "configuration.md"
ARCHITECTURE = DOCS / "architecture.md"
CITATION = REPO_ROOT / "CITATION.cff"
#: Files whose links and command mentions are checked.
DOC_FILES = [README, *sorted(DOCS.glob("*.md"))]

SECTION_CLASSES: dict[str, type] = {
    "grid": cfg.GridConfig,
    "data": cfg.DataConfig,
    "model": cfg.ModelConfig,
    "train": cfg.TrainConfig,
}
#: Presets compared in docs/configuration.md, in column order.
PRESETS = ("paracetamol", "aspirin", "ac-ala3-nhme", "stachyose", "legacy-head")
#: Heading that separates the field reference from the preset table.
PRESET_HEADING = "## Presets and `legacy-head`"
#: Section headings of README.md in the required order.
README_SECTIONS = (
    "The idea",
    "Installation",
    "Quickstart",
    "Reproducing the thesis experiments",
    "Results reported in the thesis",
    "Project layout",
    "Documentation",
    "Citation",
    "License",
)

FENCE_RE = re.compile(r"^```([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
COMMAND_RE = re.compile(r"\blrmlp[ \t]+([a-z][a-z-]*)")
INVOCATION_RE = re.compile(r"(?:^|(?<=\$\()|(?<=[\s\"']))lrmlp[ \t]+([^)|;&#\n]*)")
ROW_RE = re.compile(r"^\|\s*`([a-z_]+(?:\.[a-z_0-9]+)?)`\s*\|(.*)\|\s*$", re.M)


# --------------------------------------------------------------------------- helpers


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _blocks(text: str, languages: tuple[str, ...] | None = None) -> list[str]:
    """Bodies of the fenced code blocks of ``text`` (optionally only some languages)."""
    return [
        body
        for lang, body in FENCE_RE.findall(text)
        if languages is None or lang.strip() in languages
    ]


def _strip_code(text: str) -> str:
    """``text`` without fenced code blocks and inline code spans."""
    return INLINE_CODE_RE.sub("", FENCE_RE.sub("", text))


def _section(text: str, heading: str) -> str:
    """Body of the ``## heading`` section of a Markdown document."""
    match = re.search(rf"^## {re.escape(heading)}[ \t]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"section '## {heading}' not found"
    return match.group(1)


def _slug(heading: str) -> str:
    """GitHub anchor of a Markdown heading."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Anchors defined in a Markdown file: headings (GitHub slugs) and ``<a id>``."""
    text = FENCE_RE.sub("", _read(path))
    anchors: set[str] = set(re.findall(r"<a\s+(?:id|name)=\"([^\"]+)\"", text))
    seen: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", text, re.M):
        slug = _slug(heading)
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


def _plain(value: Any) -> Any:
    """Configuration value in the form a TOML literal parses to."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _literal(cell: str) -> Any:
    """Parse the backticked TOML literal of a table cell."""
    match = re.fullmatch(r"`([^`]+)`", cell.strip())
    assert match, f"expected a single `literal`, got {cell!r}"
    return tomllib.loads(f"v = {match.group(1)}")["v"]


def _default(field: dataclasses.Field[Any]) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _all_fields() -> list[tuple[str, dataclasses.Field[Any]]]:
    return [
        (section, field)
        for section, cls in SECTION_CLASSES.items()
        for field in dataclasses.fields(cls)
    ]


def _reference_rows() -> dict[str, list[str]]:
    """Rows of the field reference: key -> cells after the key (type, default, ...)."""
    text = _read(CONFIGURATION).split(PRESET_HEADING)[0]
    rows: dict[str, list[str]] = {}
    for key, rest in ROW_RE.findall(text):
        assert key not in rows, f"{key} is documented twice in docs/configuration.md"
        rows[key] = [cell.strip() for cell in rest.split("|")]
    return rows


def _subcommands() -> set[str]:
    parser = build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(actions) == 1
    return set(actions[0].choices)


def _invocations(path: Path) -> list[str]:
    """``lrmlp`` command lines (without the program name) in the shell blocks of a file."""
    result = []
    for body in _blocks(_read(path), ("sh", "bash", "shell")):
        for line in body.replace("\\\n", " ").splitlines():
            if line.lstrip().startswith("#"):
                continue
            result += [m.strip() for m in INVOCATION_RE.findall(line)]
    return result


# --------------------------------------------------------------------------- configuration.md


@pytest.mark.parametrize(
    ("section", "name"),
    [(s, f.name) for s, f in _all_fields()],
    ids=[f"{s}.{f.name}" for s, f in _all_fields()],
)
def test_every_config_field_is_documented(section: str, name: str) -> None:
    rows = _reference_rows()
    assert f"{section}.{name}" in rows, (
        f"`{section}.{name}` is missing from the field reference in docs/configuration.md"
    )
    assert f"`{section}.{name}`" in _read(CONFIGURATION)


def test_top_level_name_is_documented() -> None:
    fields = {f.name for f in dataclasses.fields(cfg.ExperimentConfig)}
    assert fields == {"name", *SECTION_CLASSES}
    assert "name" in _reference_rows()


def test_reference_has_no_unknown_fields() -> None:
    known = {f"{s}.{f.name}" for s, f in _all_fields()} | {"name"}
    unknown = sorted(set(_reference_rows()) - known)
    assert not unknown, f"docs/configuration.md documents unknown fields {unknown}"


@pytest.mark.parametrize(
    ("section", "field"),
    _all_fields(),
    ids=[f"{s}.{f.name}" for s, f in _all_fields()],
)
def test_documented_defaults_match_the_code(section: str, field: dataclasses.Field[Any]) -> None:
    cells = _reference_rows()[f"{section}.{field.name}"]
    assert len(cells) == 5, f"{section}.{field.name}: expected 6 columns, got {cells}"
    default_cell = cells[1]
    default = _default(field)
    if default is dataclasses.MISSING:
        assert default_cell == "required", f"{section}.{field.name} has no default"
    elif default is None:
        assert default_cell, f"{section}.{field.name}: describe the default"
    else:
        assert _literal(default_cell) == _plain(default), (
            f"{section}.{field.name}: documented default {default_cell}, code {default!r}"
        )


def test_preset_table_matches_the_configs() -> None:
    text = _read(CONFIGURATION).split(PRESET_HEADING)[1]
    header = re.search(r"^\| Key \| Default \|(.*)\|\s*$", text, re.M)
    assert header, "preset table not found"
    columns = [c.strip().strip("`") for c in header.group(1).split("|")]
    assert tuple(columns) == PRESETS
    configs = {p: cfg.load_config(REPO_ROOT / "configs" / f"{p}.toml") for p in PRESETS}
    fields = {f"{s}.{f.name}": f for s, f in _all_fields() if f.name != "sources"}
    rows = ROW_RE.findall(text)
    assert len(rows) >= 10, "preset table has too few rows"
    for key, rest in rows:
        assert key in fields, f"preset table: unknown key {key}"
        cells = [c.strip() for c in rest.split("|")]
        assert len(cells) == 1 + len(PRESETS), f"{key}: wrong number of columns"
        assert _literal(cells[0]) == _plain(_default(fields[key])), f"{key}: default"
        section, name = key.split(".")
        for preset, cell in zip(PRESETS, cells[1:], strict=True):
            value = getattr(getattr(configs[preset], section), name)
            assert _literal(cell) == _plain(value), f"{key} in {preset}: {cell} != {value!r}"


def test_preset_table_lists_every_field_that_differs() -> None:
    text = _read(CONFIGURATION).split(PRESET_HEADING)[1]
    listed = {key for key, _ in ROW_RE.findall(text)}
    skip = {"data.sources", "data.output_dir"}
    for preset in PRESETS:
        config = cfg.load_config(REPO_ROOT / "configs" / f"{preset}.toml")
        for section, field in _all_fields():
            key = f"{section}.{field.name}"
            if key in skip or field.default is dataclasses.MISSING:
                continue
            value = getattr(getattr(config, section), field.name)
            if field.name == "run_root":
                assert value == REPO_ROOT / "runs", f"{preset}: train.run_root"
                continue
            if value != _default(field):
                assert key in listed, f"{key} differs in {preset} but is not in the preset table"


# --------------------------------------------------------------------------- links


def _links() -> list[tuple[Path, str]]:
    return [
        (path, link) for path in DOC_FILES for link in LINK_RE.findall(_strip_code(_read(path)))
    ]


def test_docs_contain_links() -> None:
    assert len(_links()) > 30


@pytest.mark.parametrize(
    ("path", "link"), _links(), ids=[f"{p.name}:{link}" for p, link in _links()]
)
def test_relative_link_resolves(path: Path, link: str) -> None:
    if re.match(r"^[a-z][a-z0-9+.-]*:", link):  # http:, https:, mailto:
        return
    target, _, anchor = link.partition("#")
    resolved = (path.parent / target).resolve() if target else path
    assert resolved.exists(), f"{path.relative_to(REPO_ROOT)}: link {link!r} does not resolve"
    if anchor and resolved.suffix == ".md":
        assert anchor in _anchors(resolved), (
            f"{path.relative_to(REPO_ROOT)}: anchor #{anchor} not found in "
            f"{resolved.relative_to(REPO_ROOT)}"
        )


# --------------------------------------------------------------------------- commands


def test_every_mentioned_command_exists() -> None:
    commands = _subcommands()
    mentioned: dict[str, set[str]] = {}
    for path in DOC_FILES:
        for name in COMMAND_RE.findall(_read(path)):
            mentioned.setdefault(name, set()).add(path.name)
    unknown = {name: files for name, files in mentioned.items() if name not in commands}
    assert not unknown, f"documented commands that do not exist: {unknown}"
    readme = set(COMMAND_RE.findall(_read(README)))
    assert readme == commands, f"README does not mention {sorted(commands - readme)}"


@pytest.mark.parametrize("path", DOC_FILES, ids=[p.name for p in DOC_FILES])
def test_command_lines_parse(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)  # CLI paths are relative to the working directory
    parser = build_parser()
    for line in _invocations(path):
        argv = shlex.split(line)
        if argv in (["--help"], ["--version"]):
            assert build_parser().format_help()
            continue
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path.name}: 'lrmlp {line}' does not parse (exit {exc.code})")
        config = getattr(args, "config", None)
        if config is not None and "$" not in str(config):
            assert config.is_file(), f"{path.name}: 'lrmlp {line}': {config} not found"
            if args.command in ("prepare", "train"):
                cfg.load_config(config, args.overrides)  # raises on invalid overrides
        for attr in ("space", "input"):
            value = getattr(args, attr, None)
            if value is not None and "$" not in str(value):
                assert value.is_file(), f"{path.name}: 'lrmlp {line}': {value} not found"


def test_reproducing_covers_every_molecule() -> None:
    lines = _invocations(DOCS / "reproducing.md")
    commands = {shlex.split(line)[0] for line in lines}
    assert {"prepare", "train", "evaluate", "plot-log", "plot-weights"} <= commands
    assert any("--epoch best" in line for line in lines if line.startswith("evaluate"))
    text = _read(DOCS / "reproducing.md")
    for preset in PRESETS[:4]:
        assert f"data/raw/{preset}.npz" in text
        config = cfg.load_config(REPO_ROOT / "configs" / f"{preset}.toml")
        assert config.data.sources == (REPO_ROOT / "data" / "raw" / f"{preset}.npz",)


# --------------------------------------------------------------------------- README


def test_readme_sections_in_order() -> None:
    headings = re.findall(r"^## (.+?)\s*$", _read(README), re.M)
    assert tuple(headings) == README_SECTIONS


def test_readme_quickstart_has_one_shell_block() -> None:
    fences = FENCE_RE.findall(_section(_read(README), "Quickstart"))
    assert [lang.strip() for lang, _ in fences] == ["sh"]
    lines = _invocations(README)
    for command in ("prepare", "train", "evaluate", "predict"):
        assert any(line.startswith(command) for line in lines), command
    assert any("--set train.epochs=2" in line for line in lines)


def test_readme_reports_thesis_results() -> None:
    text = _section(_read(README), "Results reported in the thesis")
    assert "reported in the thesis" in text
    for value in ("2.465", "3.024", "1.670", "1.657", "4.039", "4.087", "5.841", "5.861"):
        assert value in text, f"Tab. 5.3 value {value} missing"
    for value in ("0.113", "0.047", "0.391", "4.002", "0.155"):
        assert value in text, f"Tab. 6.1 value {value} missing"
    assert "Tab. 5.3, p. 53" in text and "Tab. 6.1, p. 56" in text
    # The conclusion of the thesis (p. 53, p. 56), not a stronger claim.
    flat = " ".join(text.split())
    assert "does **not significantly improve** the predictions (p. 53)" in flat
    assert "sGDML gives significantly better predictions for all four molecules" in flat
    assert "clearly helps" not in flat


def test_readme_license_section() -> None:
    text = _section(_read(README), "License")
    assert "not yet specified" in text and "all rights reserved by the author" in text
    assert not any((REPO_ROOT / name).exists() for name in ("LICENSE", "LICENSE.md", "COPYING"))


def test_readme_citation_is_bibtex_mastersthesis() -> None:
    text = _section(_read(README), "Citation")
    (bibtex,) = _blocks(text, ("bibtex",))
    assert bibtex.startswith("@mastersthesis{")
    for key in ("author", "title", "school", "year"):
        assert re.search(rf"^\s*{key}\s*=", bibtex, re.M), key
    assert "CITATION.cff" in text


# --------------------------------------------------------------------------- CITATION.cff


def test_citation_cff_required_keys() -> None:
    text = _read(CITATION)
    assert re.search(r"^cff-version:\s*1\.2\.0\s*$", text, re.M)
    assert re.search(r"^title:\s*\S", text, re.M)
    assert re.search(r"^message:\s*\S", text, re.M)
    assert re.search(r"^authors:\s*\n\s+- family-names:\s*\S", text, re.M)
    assert re.search(r"^type:\s*software\s*$", text, re.M)
    assert re.search(r"^repository-code:\s*\S", text, re.M)
    assert re.search(r"^preferred-citation:\s*\n(?:\s+.*\n)*?\s+type:\s*thesis\s*$", text, re.M)
    assert "@" not in text and "email" not in text, "CITATION.cff must not contain emails"


def test_citation_cff_title_matches_readme() -> None:
    title = "Machine Learning Potentials with Long Range Interaction"
    assert title in _read(CITATION)
    assert title in _section(_read(README), "Citation")


# --------------------------------------------------------------------------- runnable examples


def _env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env["MPLBACKEND"] = "Agg"
    return env


@pytest.mark.slow
def test_readme_quickstart_runs(tmp_path: Path) -> None:
    (block,) = _blocks(_section(_read(README), "Quickstart"), ("sh",))
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", block],
        cwd=REPO_ROOT,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"quickstart failed:\n{proc.stdout}\n{proc.stderr}"
    out = Path(proc.stdout.strip().splitlines()[-1].removeprefix("outputs in ")).resolve()
    assert out.is_relative_to(tmp_path.resolve()), f"outputs not in the temp dir: {out}"
    with np.load(out / "predictions.npz") as data:
        assert data["E"].shape == (60,)
        assert np.all(np.isfinite(data["E"]))
    assert (out / "run" / "best.json").is_file()
    assert list((out / "plots").glob("*.pdf"))


@pytest.mark.slow
def test_architecture_example_runs(tmp_path: Path) -> None:
    (code,) = _blocks(_read(ARCHITECTURE), ("python",))
    # Run it as a saved script, the way a reader does: with "python -c" spawn does
    # not re-import the main module, which would hide a missing __main__ guard.
    script = tmp_path / "example.py"
    script.write_text(code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"example failed:\n{proc.stdout}\n{proc.stderr}"
    assert "test MAE:" in proc.stdout and "energies:" in proc.stdout
