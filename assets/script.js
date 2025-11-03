document.addEventListener('DOMContentLoaded', () => {
  // Rendu des formules mathématiques
  document.querySelectorAll('.math').forEach(element => {
    const texContent = element.textContent.trim();
    const mathContent = texContent.replace(/^\$\$|\$\$$/g, '');
    try {
      katex.render(mathContent, element, {
        displayMode: true,
        throwOnError: false,
        strict: false,
        output: 'html'
      });
    } catch (e) {
      console.error('Erreur KaTeX:', e);
      element.textContent = texContent;
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