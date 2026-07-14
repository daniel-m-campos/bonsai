"""Rewrite links that escape docs/ into GitHub blob URLs at build time.

The guide's code links (../../src/split.cpp) are load-bearing — "here are the
real 50 lines" — and must work both on GitHub (relative, from the file's repo
location) and on the site (where the target isn't a documentation page). The
markdown keeps the relative form; only the rendered site gets absolute URLs.
"""

from __future__ import annotations

import posixpath
import re

REPO_BLOB = "https://github.com/daniel-m-campos/bonsai/blob/main/"
REPO_RAW = "https://raw.githubusercontent.com/daniel-m-campos/bonsai/main/"
LINK = re.compile(r"(!?\]\()(\.\.?/[^)#\s]*)([^)]*\))")


def on_page_markdown(markdown, page, config, files):
    page_dir = posixpath.dirname(page.file.src_path)  # relative to docs/

    def rewrite(m):
        target = posixpath.normpath(posixpath.join("docs", page_dir, m.group(2)))
        if target.startswith("docs/"):
            return m.group(0)  # stays inside the site; leave it alone
        # images need raw bytes; the blob URL serves an HTML page
        base = REPO_RAW if m.group(1).startswith("!") else REPO_BLOB
        return m.group(1) + base + target + m.group(3)

    return LINK.sub(rewrite, markdown)
