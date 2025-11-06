---
id: intro
title: "Introduction à l'auto-différentiation. "
sidebar_position: 1
---

Bonjour ! 

Je vais vous détailler comment j'ai pu implémenter un moteur d'autodiff from scratch. 
Si vous le souhaitez, voici le [*Lien Github*](https://github.com/Hykrow/engine_rs) pour accéder à la repo.

Ce projet vient initialement de la [vidéo d'Andrej Kaparthy](https://www.youtube.com/watch?v=VMj-3S1tku0) sur le sujet. Mais j'ai remarqué que pour généraliser à des Tenseurs, il n'y avait que très peu de ressources **complètes** qui expliquaient tout bien comme je le voulais. Le but est de référencer ce qui existe déjà, et d'écrire des détails afin d'arriver à quelquechose de vraiment complet.

Ce tutoriel ira en profondeur dans les définitions mathématiques ainsi que dans le code. Il suppose cependant un niveau en mathématiques L2 ou plus, et une idée minimale de ce que fait un tel moteur (style PyTorch, Tensorflow, JAX).

J'ai choisi Rust car je trouve que les langages typés nous forcent à mieux comprendre ce qu'on écrit, et, comme ce langage est pointilleux avec la mémoire, je trouve que je dois encore mieux comprendre ce que j'écris pour que ça fonctionne.