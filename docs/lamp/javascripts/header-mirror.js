// Clone la barre du site sur TOUTES les pages de la doc, à l'identique.
// 1) Essaie d'abord /partials/header.html (si tu le crées).
// 2) Sinon, va chercher l'entête de la page d'accueil (/) et l'injecte.
// Le tout dans un Shadow DOM => aucun style ne pollue la doc.

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

    // 1) Récupérer le HTML de la barre
    let headerHTML = await fetchText('/partials/header.html'); // option recommandé si tu crées ce fichier
    if (!headerHTML) {
      const home = await fetchText('/');
      if (home) {
        const doc = new DOMParser().parseFromString(home, 'text/html');
        // Essaie plusieurs sélecteurs probables
        const sel = '#site-header, header[role="banner"], header.site, header';
        const el = doc.querySelector(sel);
        if (el) headerHTML = el.outerHTML;
      }
    }

    // 2) Récupérer les CSS du site à appliquer SEULEMENT à la barre
    //    (injectées DANS le Shadow DOM => pas de fuite)
    const cssPieces = [];
    const css1 = await fetchText('/style.css');       // si tu as style.css à la racine
    if (css1) cssPieces.push(css1);
    const css2 = await fetchText('/assets/style.css'); // si tu as aussi /assets/style.css
    if (css2) cssPieces.push(css2);
    const cssAll = cssPieces.join('\n');

    // 3) Injecter HTML + CSS dans le Shadow
    shadow.innerHTML = `
      <style>
        /* Reset léger du host pour éviter les interférences */
        :host { 
          all: initial; 
          display: block; 
          position: fixed; 
          top: 0; 
          left: 0; 
          right: 0; 
          z-index: 1002;
          font-size: 16px;
          line-height: 1.6;
          background: var(--bg);
          color: var(--text);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .wrap { 
          all: unset; 
          display: block;
          padding: 2rem 1rem;
        }
        /* Réapplique les styles exacts du site */
        .container {
          max-width: 1000px;
          margin: 0 auto;
        }
        .nav {
          margin-bottom: 4rem;
          padding-bottom: 1rem;
          border-bottom: 1px solid var(--border);
        }
      </style>
      <div class="wrap">${headerHTML || ''}</div>
    `;
    if (cssAll) {
      const style = document.createElement('style');
      style.textContent = cssAll;
      shadow.appendChild(style);
    }

    // 4) Calculer la hauteur et pousser le contenu de la doc en dessous
    const update = () => {
      // hauteur totale de notre host (incluant la barre clonée)
      const h = this.getBoundingClientRect().height || 0;
      document.documentElement.style.setProperty('--site-header-h', `${h}px`);
    };
    // attendre un frame pour que la barre se mette en place
    requestAnimationFrame(update);
    window.addEventListener('resize', update);
  }
}

customElements.define('site-header-mirror', SiteHeaderMirror);