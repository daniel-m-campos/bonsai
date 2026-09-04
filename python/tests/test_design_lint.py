"""Tests for the design ratchet's instrument in scripts/design_lint.py."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import design_lint  # noqa: E402

TABLE = "".join(f'    "key{i}": "value{i}",\n' for i in range(12))
BLOCK = ("def walk(items):\n"
         "    total = 0\n"
         "    for item in items:\n"
         "        if item.weight > LIMIT:\n"
         "            continue\n"
         "        total += item.weight * item.count\n"
         "    return total\n")


def _tree(tmp_path: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    """Write files under tmp_path by relative path and return the root."""
    for rel, body in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return tmp_path


def test_a_literal_only_row_sits_outside_the_narrative():
    """A line made of literals and punctuation is a table row, so two tables
    sharing a shape are not two copies of code."""
    assert design_lint._is_bare_literal('    "reg:linear": "mse",')
    assert design_lint._is_bare_literal("    {0, 1, 2},")
    assert design_lint._is_bare_literal("    'a', 'b',")
    assert design_lint._is_bare_literal('                  "Model"),')
    assert not design_lint._is_bare_literal('    Panel("a", "b",')
    assert not design_lint._is_bare_literal('    x = "a"')
    assert not design_lint._is_bare_literal("    }")


def test_clone_windows_count_copied_code_and_not_a_shared_table(tmp_path):
    """The same twelve-row table in two files is data; the same seven-line
    function in two files is the one clone the metric reports."""
    root = _tree(tmp_path, {
        "python/bonsai/a.py": f"A = {{\n{TABLE}}}\n\n{BLOCK}",
        "python/bonsai/b.py": f"B = {{\n{TABLE}}}\n\n{BLOCK}",
    })
    files = design_lint._production_files(root)

    reading = design_lint._clone_windows(files, root, blind_to_identifiers=False)
    assert reading.value == 2
    assert reading.top_sites == [("python/bonsai/b.py", 2)]


def test_production_files_reach_scripts_and_skip_the_rest(tmp_path):
    """The gate scripts are production code the ratchet reads; generated
    files, build trees, shell, and docs are not."""
    root = _tree(tmp_path, {
        "include/a.hpp": "", "src/b.cpp": "", "python/bonsai/c.py": "",
        "python/bonsai/_params.py": "", "scripts/d.py": "",
        "scripts/e.sh": "", "scripts/build/f.py": "", "docs/g.py": "",
        "python/tests/h.py": "",
    })

    assert [str(f.relative_to(root)) for f in design_lint._production_files(root)] == [
        "include/a.hpp", "python/bonsai/c.py", "scripts/d.py", "src/b.cpp"]
