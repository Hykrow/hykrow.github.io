---
title: "Formalisme mathématique - Gradients"
sidebar_position: 4
---

## Définition



$$
\text{Soit } f : \mathcal{T} \to \mathbb{R} \text{ une fonction différentiable, où } 
\mathcal{T} \text{ est l'espace des tenseurs.}
$$

$$
\text{Le gradient de } f \text{ en } A \in \mathcal{T} \text{ est le tenseur } 
\nabla f(A) \in \mathcal{T}
\text{ défini par :}
$$


$$
\forall\, \mathrm{d}A \in \mathcal{T}, \qquad 
\mathrm{d}f(A) = \langle \nabla f(A),\, \mathrm{d}A \rangle
$$


Beaucoup de la littérature au sujet des moteurs d'autodiff utilisent systématiquement la chain rule pour justifier leur calcul de gradients. Mais ce n'est pas nécessaire (et trop compliqué) une fois qu'on a posé cette définition ! 

Expliquer quon veut calculer le gradient d'un element par rapport à un autre
## Opérations usuelles

### Addition

### Multiplication élément par élément (Hadamard)

### Multiplication 

### Application de fonction