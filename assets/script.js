document.addEventListener('DOMContentLoaded', () => {
  // Configuration du rendu LaTeX
  renderMathInElement(document.body, {
    delimiters: [
      {left: "\\[", right: "\\]", display: true},
      {left: "\\(", right: "\\)", display: false},
      {left: "$", right: "$", display: false},
      {left: "$$", right: "$$", display: true}
    ],
    throwOnError: false,
    strict: false,
    trust: true,
    macros: {
      "\\mathbb": "\\mathbb",
      "\\mathcal": "\\mathcal"
    }
  });

  // Gestion du thème
  const root = document.documentElement;
  const key = "nf-theme";
  const btn = document.getElementById("themeToggle");
  
  function apply(theme) {
    if (theme === "light") root.classList.add("light");
    else root.classList.remove("light");
    if (btn) btn.textContent = theme === "light" ? "☀︎" : "☾";
  }

  const saved = localStorage.getItem(key);
  if (saved) apply(saved);

  if (btn) {
    btn.addEventListener("click", () => {
      const isLight = root.classList.contains("light");
      const next = isLight ? "dark" : "light";
      apply(next);
      localStorage.setItem(key, next);
    });
  }
});