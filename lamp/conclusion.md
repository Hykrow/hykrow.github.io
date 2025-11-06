---
title: "Conclusion"
sidebar_position: 10
---
J'ai entrainé le modèle pendant 20 minutes (5 epochs, avec 6000 train et 600 test) et j'ai obtenu une accuracy de environ 60-70%. Le problème avec le code est qu'il est très simple afin d'être lisible, mais il est donc très lent. Il faut paraléliser les opérations sur le CPU (possible car `Arc` le permet), voire aller jusqu'aux multiplications via GPU. Une si faible accuracy peut s'expliquer par le fait que j'ai train uniquement sur 10% du dataset.. (et avec pas énormément d'epochs)

## Pistes d'amélioration & autres

Je compte donc implémenter la parallélisation CPU, ainsi que d'autres couches comme de la régularisaition, convolution etc... Peut-être même multiplication sur GPU si j'ai vraiment le temps; mais ce post ne sera pas nécessairement remis à jour. Donc n'hésitez pas à regarder le [GitHub](https://github.com/Hykrow/engine_rs) si vous le souhaitez.

N'hésitez pas à star la repo si ca vous plaît / vous a plu! Et a contribuer (au site ou a la repo directement) si vous le voulez!