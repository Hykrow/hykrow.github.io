// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Noam — Blog & Projets',
  url: 'https://hykrow.github.io',
  baseUrl: '/',
  favicon: 'img/favicon.ico',

  organizationName: 'hykrow',
  projectName: 'hykrow.github.io',
  deploymentBranch: 'gh-pages',

  i18n: { defaultLocale: 'fr', locales: ['fr'] },
  onBrokenLinks: 'throw',

  // ⚠️ D3.9: déplace l’option de markdown "broken links" ici
  markdown: { hooks: { onBrokenMarkdownLinks: 'warn' } },

    // KaTeX + Fonts (Inter pour titres, IBM Plex Mono pour le corps façon "Cactus")
  stylesheets: [
    { href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css', type: 'text/css' },
    { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap', rel: 'stylesheet' },
     // ← on force ce CSS global
  ],




  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          showLastUpdateTime: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: { customCss: require.resolve('./src/css/custom.css') },
      }),
    ],
  ],

  themeConfig: {
    colorMode: { respectPrefersColorScheme: true, defaultMode: 'light' },
    navbar: {
      title: 'Accueil',
      items: [
        //{ to: '/blog', label: 'Blog', position: 'left' },
        { to: '/projects', label: 'Projets', position: 'right'},
        { to: '/cv', label: 'CV', position: 'right' },
        { to: '/contact', label: 'Contact', position: 'right' },
        { to: '/interets', label: 'Intérêts', position: 'right' },
      ],
    },
    tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 4 },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
  plugins: [
  [
    '@docusaurus/plugin-content-docs',
    {
      id: 'lamp',                         // ID unique de l'instance
      path: 'lamp',                       // dossier source des MD
      routeBasePath: 'lamp',              // URL = /lamp
      sidebarPath: require.resolve('./sidebarsLamp.js'),
      // mêmes options Markdown que tes docs/blog (maths etc.)
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
      showLastUpdateTime: true,
    },
  ],
],

};

export default config;
