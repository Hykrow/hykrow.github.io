class SiteHeaderMirror extends HTMLElement {
  async connectedCallback() {
    try {
      // Charge le header depuis /partials/header.html
      const res = await fetch('/partials/header.html', { cache: 'no-store' });
      if (res.ok) {
        const html = await res.text();
        this.innerHTML = html;
      }
    } catch (e) {
      console.error('Erreur lors du chargement du header:', e);
    }
  }
}

customElements.define('site-header-mirror', SiteHeaderMirror);