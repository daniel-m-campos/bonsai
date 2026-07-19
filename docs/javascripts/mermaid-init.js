// Render mermaid fences client-side. The theme ships mermaid CSS but no
// loader, so the library loads explicitly, the same pattern as katex.js.
// document$ re-fires on instant navigation; data-processed guards reruns.
document$.subscribe(() => {
  if (!window.mermaid) return;
  window.mermaid.initialize({ startOnLoad: false, theme: "neutral" });
  window.mermaid.run({ querySelector: ".mermaid:not([data-processed])" });
});
