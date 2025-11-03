import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/blog',
    component: ComponentCreator('/blog', '316'),
    exact: true
  },
  {
    path: '/blog/2025/11/03/moteur-autodiff',
    component: ComponentCreator('/blog/2025/11/03/moteur-autodiff', '4e7'),
    exact: true
  },
  {
    path: '/blog/archive',
    component: ComponentCreator('/blog/archive', '182'),
    exact: true
  },
  {
    path: '/blog/tags',
    component: ComponentCreator('/blog/tags', '287'),
    exact: true
  },
  {
    path: '/blog/tags/autodiff',
    component: ComponentCreator('/blog/tags/autodiff', 'bff'),
    exact: true
  },
  {
    path: '/blog/tags/rust',
    component: ComponentCreator('/blog/tags/rust', '3cc'),
    exact: true
  },
  {
    path: '/blog/tags/tensor',
    component: ComponentCreator('/blog/tags/tensor', '3f2'),
    exact: true
  },
  {
    path: '/blog/tags/vjp',
    component: ComponentCreator('/blog/tags/vjp', '2da'),
    exact: true
  },
  {
    path: '/contact',
    component: ComponentCreator('/contact', 'b83'),
    exact: true
  },
  {
    path: '/cv',
    component: ComponentCreator('/cv', '821'),
    exact: true
  },
  {
    path: '/interets',
    component: ComponentCreator('/interets', '68d'),
    exact: true
  },
  {
    path: '/projects',
    component: ComponentCreator('/projects', 'e7b'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '3ef'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'e49'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '425'),
            routes: [
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '5ac'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/roadmap',
                component: ComponentCreator('/docs/roadmap', 'c40'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/lamp',
    component: ComponentCreator('/lamp', '991'),
    routes: [
      {
        path: '/lamp',
        component: ComponentCreator('/lamp', 'e1d'),
        routes: [
          {
            path: '/lamp',
            component: ComponentCreator('/lamp', 'f12'),
            routes: [
              {
                path: '/lamp/gradients',
                component: ComponentCreator('/lamp/gradients', '99d'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/graphe',
                component: ComponentCreator('/lamp/graphe', 'fe3'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/intro',
                component: ComponentCreator('/lamp/intro', '815'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/tenseurs',
                component: ComponentCreator('/lamp/tenseurs', '4be'),
                exact: true,
                sidebar: "lampSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', 'fd5'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
