// KaTeX auto-render for arithmatex generic output (```math fences render as
// <div class="arithmatex">\[...\]</div>). Single-dollar inline math is
// deliberately not configured; use \( ... \) if inline math is ever needed.
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: true },
    ],
  });
});
