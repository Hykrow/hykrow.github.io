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

  i18n: { defaultLocale: 'fr', locales: ['fr', 'en'] },
  onBrokenLinks: 'throw',

  markdown: { hooks: { onBrokenMarkdownLinks: 'warn' }, mermaid: true }, // 🧩 active Mermaid dans Markdown

  stylesheets: [
    { href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css', type: 'text/css' },
    { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap', rel: 'stylesheet' },
  ],

  themes: ['@docusaurus/theme-mermaid'], // 🧩 ajoute le thème Mermaid ici

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

    // 🧩 style Mermaid (clair/sombre)
  mermaid: {
    theme: { light: 'neutral', dark: 'dark' },
  },

    navbar: {
      title: 'Accueil',
      items: [
       { type: 'localeDropdown', position: 'right' },
       { to: '/projects', label: 'Projets', position: 'right'},
       { to: '/cv', label: 'CV', position: 'right' },
       { to: '/contact', label: 'Contact', position: 'right' },
        { to: '/interets', label: 'Intérêts', position: 'right', locale: 'fr' },
  { to: '/interets', label: 'Interests', position: 'right', locale: 'en' },
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
        id: 'lamp',
        path: 'lamp',
        routeBasePath: 'lamp',
        sidebarPath: require.resolve('./sidebarsLamp.js'),
        remarkPlugins: [remarkMath],
        rehypePlugins: [rehypeKatex],
        showLastUpdateTime: true,
      },
    ],
  ],
};

export default config;
