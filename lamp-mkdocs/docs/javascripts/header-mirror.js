// Copie la barre du site dans le flux (non-sticky) + Fallback CSS horizontal.
class SiteHeaderMirror extends HTMLElement {
  async connectedCallback() {
    const shadow = this.attachShadow({ mode: 'open' });

    const fetchText = async (url) => {
      try {
        const r = await fetch(url, { cache: 'no-store' });
        if (r.ok) return await r.text();
      } catch (_) {}
      return null;
    };

    // 1) Récupère le HTML exact de la barre
    let headerHTML = await fetchText('/partials/header.html'); // si dispo
    if (!headerHTML) {
      const home = await fetchText('/');
      if (home) {
        const doc = new DOMParser().parseFromString(home, 'text/html');
        const el =
          doc.querySelector('#site-header') ||
          doc.querySelector('header[role="banner"]') ||
          doc.querySelector('header.site') ||
          doc.querySelector('header');
        if (el) headerHTML = el.outerHTML;
      }
    }

    // 2) Fallback CSS (mini) pour forcer une nav horizontale si aucune CSS du site
    const fallbackCSS = `
      :host { all: initial; display: block; }
      .wrap { all: unset; display: block; }
      /* Enlève les puces et marges par défaut */
      .wrap ul, .wrap ol { list-style: none; margin: 0; padding: 0; }
      /* Nav horizontale dans les cas courants */
      .wrap nav ul,
      .wrap .nav,
      .wrap .navbar,
      .wrap .menu,
      .wrap .links {
        display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
      }
      .wrap a { text-decoration: none; }
    `;

    // 3) CSS du site (appliquée dans le Shadow, donc pas de fuite)
    const siteCSS = [];
    const cssRoot = await fetchText('/style.css');        if (cssRoot)   siteCSS.push(cssRoot);
    const cssAssets = await fetchText('/assets/style.css'); if (cssAssets) siteCSS.push(cssAssets);
    const cssAll = siteCSS.join('\n');

    // 4) Injection
    shadow.innerHTML = `
      <style>${fallbackCSS}</style>
      <div class="wrap">${headerHTML || ''}</div>
    `;
    if (cssAll) {
      const style = document.createElement('style');
      style.textContent = cssAll;          // le CSS du site override le fallback
      shadow.appendChild(style);
    }
  }
}
customElements.define('site-header-mirror', SiteHeaderMirror);
