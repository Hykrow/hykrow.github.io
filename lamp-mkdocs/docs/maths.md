---
title: Maths
---

# Maths

Description des choix mathématiques : notation, autodiff (VJP), stabilité numérique, broadcasting.

On présente ici les formules utilisées et la correspondance avec les primitives du moteur.

Formule illustrative :

$$\frac{\partial \mathcal{L}}{\partial \theta} = \mathbb{E}_{x \sim \mathcal{D}} \left[\nabla_\theta f_\theta(x) \cdot \nabla_f \mathcal{L}(f_\theta(x))\right]$$
