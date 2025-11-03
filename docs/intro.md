---
sidebar_position: 1
title: "Introduction au moteur autodiff"
---

Bienvenue ! Ici je documente la conception de mon **moteur autodiff en Rust**.

## Idée générale
- Graphe acyclique dirigé (DAG)
- Fermetures **VJP**
- Broadcasting numpy-like

### Équation clé (KaTeX)
$$
\frac{\partial \mathcal{L}}{\partial x} =
\left(\frac{\partial y}{\partial x}\right)^\top
\frac{\partial \mathcal{L}}{\partial y}.
$$
