// Thème light/dark avec persistance.
(function() {
  const key = "nf-theme";
  const root = document.documentElement;
  function apply(theme) {
    if (theme === "light") root.classList.add("light");
    else root.classList.remove("light");
  }
  const saved = localStorage.getItem(key);
  if (saved) apply(saved);

  const btn = document.getElementById("themeToggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const isLight = root.classList.contains("light");
      const next = isLight ? "dark" : "light";
      apply(next);
      localStorage.setItem(key, next);
      btn.textContent = next === "light" ? "☀︎" : "☾";
    });
  }
})();