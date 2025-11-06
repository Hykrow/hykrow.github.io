import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
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
    path: '/solutions',
    component: ComponentCreator('/solutions', '619'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'e45'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'cc7'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '885'),
            routes: [
              {
                path: '/docs/aa',
                component: ComponentCreator('/docs/aa', 'e12'),
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
    component: ComponentCreator('/lamp', '4ba'),
    routes: [
      {
        path: '/lamp',
        component: ComponentCreator('/lamp', '427'),
        routes: [
          {
            path: '/lamp',
            component: ComponentCreator('/lamp', '80a'),
            routes: [
              {
                path: '/lamp/conclusion',
                component: ComponentCreator('/lamp/conclusion', 'a87'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/construction',
                component: ComponentCreator('/lamp/construction', 'fb5'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/gradients',
                component: ComponentCreator('/lamp/gradients', '502'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/graphe',
                component: ComponentCreator('/lamp/graphe', '2fd'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/intro',
                component: ComponentCreator('/lamp/intro', '1ab'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/operations',
                component: ComponentCreator('/lamp/operations', '433'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/tenseurs_impl',
                component: ComponentCreator('/lamp/tenseurs_impl', 'e5f'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/tenseurs_math',
                component: ComponentCreator('/lamp/tenseurs_math', 'ad2'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/train',
                component: ComponentCreator('/lamp/train', 'd2c'),
                exact: true,
                sidebar: "lampSidebar"
              },
              {
                path: '/lamp/utils',
                component: ComponentCreator('/lamp/utils', '240'),
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
