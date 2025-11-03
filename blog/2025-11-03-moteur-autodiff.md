---
title: "Moteur autodiff en Rust : design, VJP et perf"
tags: [rust, autodiff, vjp, tensor]
---

> Vue d’ensemble du DAG, des **VJP**, du **broadcasting**, et de la perf CPU.

### Modèle de calcul
Chaque `Node` stocke `value`, `parents_id` et une fermeture **VJP**.
