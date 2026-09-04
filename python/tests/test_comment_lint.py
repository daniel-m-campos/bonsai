"""Tests for the comment policy gate in scripts/comment_lint.py."""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import comment_lint  # noqa: E402


def _source(tmp_path: pathlib.Path, rel: str, body: str) -> pathlib.Path:
    """Write body to rel under tmp_path and return the path."""
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_perf_with_a_number_in_the_block_passes(tmp_path):
    """A measurement inside the block satisfies the perf admission test."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// perf: measured at 3.2 ms on the L40S\n"
                   "void fast();\n")
    assert comment_lint.block_findings(path) == []


def test_perf_with_the_number_in_the_declaration_passes(tmp_path):
    """The measured constant itself counts as the number."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// perf: the cache line width sets this stride\n"
                   "constexpr int kStride = 64;\n")
    assert comment_lint.block_findings(path) == []


def test_perf_without_any_number_is_a_finding(tmp_path):
    """A hardware claim with no number fails the perf admission test."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// perf: this is faster on modern hardware\n"
                   "void slow();\n")
    assert comment_lint.block_findings(path) == [
        (1, "perf: carries no number, so it is not a measurement:"
            " // perf: this is faster on modern hardware"),
    ]


def test_perf_finding_truncates_the_quoted_line(tmp_path):
    """The perf message quotes the first 52 characters of the tag line."""
    path = _source(
        tmp_path, "src/other/a.cpp",
        "// perf: the wide form beats the narrow one on every host we measured\n"
        "void wide();\n")
    assert comment_lint.block_findings(path) == [
        (1, "perf: carries no number, so it is not a measurement:"
            " // perf: the wide form beats the narrow one on every"),
    ]


def test_perf_block_at_end_of_file_has_no_declaration(tmp_path):
    """A block with nothing under it looks at an empty declaration."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "void f();\n"
                   "// perf: nothing measured here\n")
    assert comment_lint.block_findings(path) == [
        (2, "perf: carries no number, so it is not a measurement:"
            " // perf: nothing measured here"),
    ]


def test_sync_naming_a_construct_the_code_uses_passes(tmp_path):
    """A sync tag naming a primitive the file really uses is admitted."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// sync: the mutex guards the shared counter\n"
                   "void guard();\n"
                   "std::mutex m;\n")
    assert comment_lint.block_findings(path) == []


def test_sync_naming_a_construct_the_code_lacks_is_a_finding(tmp_path):
    """Naming a primitive absent from the code fails the sync test."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// sync: the atomic counter must not tear\n"
                   "void tear();\n")
    assert comment_lint.block_findings(path) == [
        (1, "sync: names no construct this file uses:"
            " // sync: the atomic counter must not tear"),
    ]


def test_sync_naming_nothing_is_a_finding(tmp_path):
    """Prose with no primitive in it is not a sync constraint."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// sync: ordering matters here for correctness\n"
                   "void order();\n")
    assert comment_lint.block_findings(path) == [
        (1, "sync: names no construct this file uses:"
            " // sync: ordering matters here for correctness"),
    ]


def test_ffi_naming_a_boundary_construct_in_src_python_passes(tmp_path):
    """An ffi tag under src/python naming a used term is admitted."""
    path = _source(tmp_path, "src/python/a.cpp",
                   "// ffi: the capsule owns the buffer past the call\n"
                   "void own();\n"
                   "void* capsule_ptr;\n")
    assert comment_lint.block_findings(path) == []


def test_ffi_naming_no_boundary_construct_is_a_finding(tmp_path):
    """An ffi tag that names no boundary term fails even under src/python."""
    path = _source(tmp_path, "src/python/a.cpp",
                   "// ffi: the boundary must stay tidy\n"
                   "void tidy();\n")
    assert comment_lint.block_findings(path) == [
        (1, "ffi: names no boundary construct this file uses:"
            " // ffi: the boundary must stay tidy"),
    ]


def test_ffi_outside_src_python_is_a_finding(tmp_path):
    """Only src/python may carry an ffi tag, however well it is named."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// ffi: the capsule owns the buffer past the call\n"
                   "void own();\n"
                   "void* capsule_ptr;\n")
    assert comment_lint.block_findings(path) == [
        (1, "ffi: only src/python/ may carry this tag:"
            " // ffi: the capsule owns the buffer past the call"),
    ]


def test_untagged_block_reports_one_finding_per_line(tmp_path):
    """Every line of an untagged block is reported on its own."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// this explains the loop\n"
                   "// and keeps explaining it\n"
                   "void loop();\n")
    assert comment_lint.block_findings(path) == [
        (1, "// this explains the loop"),
        (2, "// and keeps explaining it"),
    ]


def test_untagged_finding_truncates_at_eighty_characters(tmp_path):
    """An untagged line is quoted to its first 80 characters."""
    path = _source(
        tmp_path, "src/other/a.cpp",
        "// this comment explains the loop at length and keeps on explaining"
        " well past eighty characters\n"
        "void loop();\n")
    assert comment_lint.block_findings(path) == [
        (1, "// this comment explains the loop at length and keeps on"
            " explaining well past ei"),
    ]


def test_structural_lines_pass(tmp_path):
    """NOLINT, clang-format toggles and namespace labels are not comments."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "// NOLINTNEXTLINE(bugprone-x)\n"
                   "void a();\n"
                   "// clang-format off\n"
                   "void b();\n"
                   "// namespace foo\n"
                   "namespace foo {\n"
                   "}  // namespace\n")
    assert comment_lint.block_findings(path) == []


def test_trailing_comment_after_code_is_a_finding(tmp_path):
    """A comment sharing a line with code is still a comment."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "int x = 1;  // set the width\n")
    assert comment_lint.block_findings(path) == [(1, "// set the width")]


def test_trailing_tagged_comment_after_code_passes(tmp_path):
    """A tagged trailing comment is admitted without the block tests."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "int x = 64;  // perf: one cache line\n")
    assert comment_lint.block_findings(path) == []


def test_double_slash_inside_a_string_is_not_a_comment(tmp_path):
    """A quote before the slashes takes the line out of the trailing check."""
    path = _source(tmp_path, "src/other/a.cpp",
                   'const char* url = "http://example.com";\n')
    assert comment_lint.block_findings(path) == []


def test_a_trailing_comment_past_a_string_is_a_finding(tmp_path):
    """A literal earlier on the line does not hide the comment after it."""
    path = _source(tmp_path, "src/other/a.cpp",
                   'const char* s = "a";  // this explains the string\n')
    assert comment_lint.block_findings(path) == [
        (1, "// this explains the string"),
    ]


def test_slashes_inside_literals_are_not_comments(tmp_path):
    """Quotes, escapes, and raw strings all keep their slashes."""
    path = _source(tmp_path, "src/other/a.cpp",
                   "char c = '\"'; int a = 1; // after a char literal\n"
                   'auto s = "esc \\" // still inside"; int b = 2;\n'
                   'auto r = R"(a "quoted" http://x)"; int d = 3;\n'
                   "char q = '//'; int e = 4;\n")
    assert comment_lint.block_findings(path) == [
        (1, "// after a char literal"),
    ]


def test_ffi_needs_src_directly_over_python(tmp_path):
    """A python component elsewhere in the path does not admit the tag."""
    path = _source(tmp_path, "python/src/a.cpp",
                   "// ffi: the capsule owns the buffer past the call\n"
                   "void* capsule_ptr;\n")
    assert comment_lint.block_findings(path) == [
        (1, "ffi: only src/python/ may carry this tag:"
            " // ffi: the capsule owns the buffer past the call"),
    ]
