"""scripts/theme_check.py: a block or class the installed Material lacks is
reported, and the live theme passes against the installed one."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("theme_check", REPO / "scripts/theme_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


theme_check = _load()


def test_blocks_the_base_template_lacks_are_reported(tmp_path: pathlib.Path):
    base = tmp_path / "base.html"
    base.write_text("{% block tabs %}{% endblock %}{%- block content -%}{% endblock %}")
    overrides = tmp_path / "overrides"
    overrides.mkdir()
    (overrides / "home.html").write_text(
        "{% block tabs %}{% block hero %}{% endblock %}{% endblock %}"
    )
    assert theme_check.base_blocks(base) == {"tabs", "content"}
    assert theme_check.override_blocks(overrides) - theme_check.base_blocks(base) == {"hero"}


def test_classes_the_bundled_stylesheet_lacks_are_reported(tmp_path: pathlib.Path):
    templates = tmp_path / "templates" / "assets"
    templates.mkdir(parents=True)
    (templates / "main.min.css").write_text(".md-grid{max-width:61rem}.md-typeset table{}")
    (tmp_path / "templates" / "base.html").write_text('<div class="md-header md-grid">')
    sheet = tmp_path / "bonsai.css"
    sheet.write_text(".md-grid{} .md-typeset .md-nav--primary{} .bz-hero .md-header{}")
    found = theme_check.material_classes(tmp_path / "templates")
    assert theme_check.stylesheet_classes(sheet) - found == {"md-nav--primary"}


def test_the_live_theme_passes_against_the_installed_material():
    pytest.importorskip("material")
    assert theme_check.main() == 0
