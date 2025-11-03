// Barre identique à la home, insérée dans le slot header (plein écran), non-sticky.
// On privilégie /partials/header.html si tu l'ajoutes ; sinon on extrait l'en-tête de "/".

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

    // 1) HTML de la barre
    let headerHTML = await fetchText('/partials/header.html');
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

    // 2) CSS du site à appliquer DANS le Shadow DOM (pas de fuite)
    const siteCSS = [];
    const cssRoot = await fetchText('/style.css');        if (cssRoot)   siteCSS.push(cssRoot);
    const cssAssets = await fetchText('/assets/style.css'); if (cssAssets) siteCSS.push(cssAssets);
    const cssAll = siteCSS.join('\n');

    // 3) Fallback CSS ultra-minimal (uniquement si les CSS du site ne sont pas trouvées)
    const fallbackCSS = `
      :host { all: initial; display: block; }
      .wrap { all: unset; display: block; }
      .wrap nav ul, .wrap .nav, .wrap .navbar, .wrap .menu, .wrap .links {
        display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
      }
      .wrap ul, .wrap ol { list-style: none; margin: 0; padding: 0; }
      .wrap a { text-decoration: none; }
    `;

    // 4) Injection
    shadow.innerHTML = `
      <style>${fallbackCSS}</style>
      <div class="wrap">${headerHTML || ''}</div>
    `;
    if (cssAll) {
      const style = document.createElement('style');
      style.textContent = cssAll;   // Override le fallback avec TES styles exacts
      shadow.appendChild(style);
    }
  }
}

customElements.define('site-header-mirror', SiteHeaderMirror);
