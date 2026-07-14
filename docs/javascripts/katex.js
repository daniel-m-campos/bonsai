// KaTeX auto-render for arithmatex generic output (```math fences render as
// <div class="arithmatex">\[...\]</div>; inline stays $...$ if ever used).
document$.subscribe(() => {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: true },
    ],
  });
});
