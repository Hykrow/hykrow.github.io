---
title: "Formalisme mathématique - Tenseurs"
sidebar_position: 2
---

Un réseau de neurone fait ses opérations sur des tenseurs. (addition, multiplication, tanh, ...)
MODIFIER CA


## Tenseurs
On a souvent tendance à ne pas comprendre ce qu'est un tenseur initialement, mais il faut imaginer ces objets comme la généralisation des scalaires/vecteurs/matrices a $$n$$ dimensions.

### Définition
Pour la définition d'un tenseur, je renvoie à l'excellent [post de "Robot Chinwag"](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/) sur le sujet. Cependant, vous pouvez vous **arrêter avant la partie "Tensor Calculus"** (et tout ce qu'il y a après) car on va utiliser des principes différents, cette partie ne sera donc pas utile pour nous.

D'orénavant, on posera $$\mathcal{T}$$ l'espace des tenseurs, et $\mathcal{T}(N)$ celui des tenseurs d'ordre $$N$$.



Par exemple: 
$$
\text{Considérons un tenseur } 
A \in \mathbb{R}^{3 \times 2 \times 2} \subset \mathcal{T}(3).
\text{ On peut le voir comme deux matrices empilées :}
$$

$$
A =
\begin{bmatrix}
\begin{bmatrix}
a_{111} & a_{112} \\
a_{121} & a_{122}
\end{bmatrix},
\quad
\begin{bmatrix}
a_{211} & a_{212} \\
a_{221} & a_{222}
\end{bmatrix},
\quad
\begin{bmatrix}
a_{311} & a_{312} \\
a_{321} & a_{322}
\end{bmatrix}

\end{bmatrix}.
$$

$$
\text{Autrement dit, } A_{k, i, j} \text{ désigne l’élément situé à la position } (i,j)
\text{ dans la $k$-ième matrice.}
$$
On note également $$\mathcal{d}_A(i)$$ la dimension d'ordre $$i$$.
Dans ce cas, on a $$\mathcal{d}_A(1) = 3 \text{ et } \mathcal{d}_A(i) = 2 \text{ pour } i \in \{2, 3\}$$

### Transposée
Soit $$A = a_{zij}  \in \mathcal{T}(N)\text{,} N \geq 2$$.

Alors
$$
A ^{\top} =  a_{zji}
$$

Avec $$i, j$$ les indices des 2 dernières dimensions. L'indice $$z$$ représente l'indice de "batch", soit le reste des dimensions précédentes.
Si les dimensions d'un tenseur sont $$i_1, ..., i_N$$, alors il faut considérer $$z$$ comme $$i_1, ..., i_{N-2}$$.

### Squeeze, unsqueeze


### Broadcast
Un concept important pour les tenseurs est celui du broadcast. 
Il était déjà expliqué dans le [post de "Robot Chinwag"](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/), mais je vais mettre des examples afin de rendre la notion plus compréhensible.


Un exemple simple peut être lorsqu'on multiplie un vecteur par un scalaire. Cela peut se voir comme le broadcast du scalaire sur la shape du vecteur puis une multiplication élément par élément. Ex: 
$$ 
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
*
3 = 
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix}
*
\begin{bmatrix}
3 & 3 & 3
\end{bmatrix}
=
\begin{bmatrix}
1*3 & 2*3 & 3*3
\end{bmatrix} =  \begin{bmatrix}
3 & 6 & 9
\end{bmatrix} 

$$

$$
\text{où } * \text{ désigne le produit élément par élément (Hadamard)}
$$

Voici un exemple plus avancé: 
$$
\text{Soient }
A \in \mathbb{R}^{3 \times 1}
\quad\text{et}\quad
B \in \mathbb{R}^{1 \times 4}.
$$

$$
A =
\begin{bmatrix}
1 \\[2mm]
2 \\[2mm]
3
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
10 & 20 & 30 & 40
\end{bmatrix}.
$$

$$
\text{Lors d'une opération élément-par-élément (par ex. } C = A + B\text{ ), }
\text{les deux tenseurs sont \emph{broadcastés} pour obtenir :}
$$

$$
A' =
\begin{bmatrix}
1 & 1 & 1 & 1 \\[2mm]
2 & 2 & 2 & 2 \\[2mm]
3 & 3 & 3 & 3
\end{bmatrix},
\qquad
B' =
\begin{bmatrix}
10 & 20 & 30 & 40 \\[2mm]
10 & 20 & 30 & 40 \\[2mm]
10 & 20 & 30 & 40
\end{bmatrix}.
$$

$$
\text{Ainsi, }
C = A + B =
\begin{bmatrix}
11 & 21 & 31 & 41 \\[2mm]
12 & 22 & 32 & 42 \\[2mm]
13 & 23 & 33 & 43
\end{bmatrix}.
$$

Plus formellement, voici le code/ pseudo code (certaines fonction rust n'existent pas) pour broadcaster 2 Tenseurs: (c'est un détail d'implémentation, si vous avez compris l'idée du broadcast dans le post que j'ai référé précédemment, c'est OK)
```rust
// Entrée : deux tenseurs A et B
// Sortie : leurs versions broadcastés A', B', et la shape finale out_shape

// Étape 1 : calcul de la shape résultante
fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
    let rank = shape_a.len().max(shape_b.len());

    // on fait du padding à gauche en ajoutant des dimensions de taille 1 virtuelles afin que les 2 tenseurs aient le même ordre.
    let a = shape_a.expand([1]*(rank-shape_a.len()));
    let b = shape_b.expand([1]*(rank-shape_b.len()));

    (0..rank) // prend les elements entre 0 et rank-1 inclus
        .map(|i| match (a[i], b[i]) {
            (x, y) if x == y => x,
            (1, y) => y, // On "duplique" le scalaire sur l'entièreté de la dimension de b correspondante.
            (x, 1) => x, // De même, symétrique.
            _ => panic!("Incompatible shapes"),
        })
        .collect()
}


//TODO: tout bien ré ecrire en rust.

// Étape 2 : expansion d'un tenseur à la shape résultante
//  Note: cette fonctoin broadcast peut être non correcte si le tenseur passé en argument est déjà non contigu, en fonction de comment on suppose qu'elle doit fonctionner
fn broadcast_to(tensor: Tensor, out_shape: Shape) -> Tensor:
    shape = tensor.shape
    rank_diff = len(out_shape) - len(shape)
    shape = [1]*rank_diff + shape  # alignement à gauche

    broadcasted = tensor.copy()
    broadcasted.shape = out_shape
    return broadcasted

fn broadcast(tensorA: Tensor, tensorB: Tensor) -> Tensor:
    shape = broadcast_shape(tensorA.shape, tensorB.shape)
    return broadcast_to(tensorA, shape), broadcast_to(tensorB, shape)
```
On supposera maintenant que pour chaque opération (multiplication, addition, etc etc... On **broadcastera** les tenseurs afin que leur shapes correspondent si c'est possible).

### Produit scalaire

$$
\text{Soient } A, B \in \mathcal{T}(N) \times \mathcal{T}(M), \text{ t.q } \mathcal{d}_A(N-1) = \mathcal{d}_B(M-1)
$$

$$
\text{ On définit leur produit scalaire par : }
$$

$$
\langle A, B \rangle = \operatorname{Tr}(A^{\top} B)
= a_{zji}b_{zij}
$$
Si vous avez toujours du mal avec cette notation, malgré le [post de "Robot Chinwag"](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/), je vous conseille de regarder d'autres resources sur la notation d'Einstein et même sur einsum (fonction PyTorch). (En gros ici, on somme sur les indices $$z$$)

Ensuite, définissons la notion de gradient (qui n'est pas forcément immédiate pour autre chose que des scalaires, encore moins des Tenseurs )
