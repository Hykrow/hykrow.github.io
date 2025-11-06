---
title: "Mathematical formalism - Tensors"
sidebar_position: 2
---

A neural network performs operations on tensors (addition, multiplication, tanh, etc.). So, we will first focus on implementing everything correctly for tensors.


## Tensors
People often don't understand what a tensor is at first, but you can think of these objects as the generalization of scalars/vectors/matrices to $$n$$ dimensions.

### Definition
For the definition of a tensor, I refer you to the excellent [post by “Robot Chinwag”](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/) on the subject. However, you can **stop before the “Tensor Calculus” section** (and everything after it) because we will be using different principles, so this section will not be useful to us.

From now on, we will denote $$\mathcal{T}$$ as the tensor space, and $\mathcal{T}(N)$ as the tensor space of order $$N$$.



For example: 
$$
\text{Lets consider a tensor: } 
A \in \mathbb{R}^{3 \times 2 \times 2} \subset \mathcal{T}(3).
\text{ We can see it as two stacked matrices :}
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
\text{ In other words, } A_{k, i, j} \text{ denotes the element at position } (i,j)
\text{ in the $k$-th matrix.}
$$
We also denote $$\mathcal{d}_A(i)$$ as the dimension of order $$i$$.
In this case, we have $$\mathcal{d}_A(1) = 3 \text{ and } \mathcal{d}_A(i) = 2 \text{ for } i \in \{2, 3\}$$

### Transpose
Let $$A = a_{zij}  \in \mathcal{T}(N)\text{,} N \geq 2$$.

Then
$$
A ^{\top} =  a_{zji}
$$

With $$i, j$$ the indices of the last two dimensions. The index $$z$$ represents the “batch” index, i.e., the remainder of the previous dimensions.
If the dimensions of a tensor are $$i_1, ..., i_N$$, then $$z$$ must be considered as $$i_1, ..., i_{N-2}$$.




### Broadcast
An important concept for tensors is that of broadcast. 
It was already explained in the post of[Robot Chinwag](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/), but I will provide some examples to make the concept easier to understand.


A simple example is when you multiply a vector by a scalar. This can be seen as broadcasting the scalar over the shape of the vector and then multiplying element by element. Example: 

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
\text{où } * \text{ denotes the element-wise product (Hadamard)}
$$

Here is a more avanced example:
$$
\text{Let }
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
\text{During an element-by-element operation (e.g., } C = A + B\text{ ), }
\text{both tensors are \emph{broadcast} to get:}
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

You can find the code for broadcasting in the following section, if that helps you understand better.
We will now assume that for each operation (multiplication, addition, etc.), we will **broadcast** the tensors so that their shapes match if possible.


### Scalar product

$$
\text{Let } A, B \in \mathcal{T}(N) \times \mathcal{T}(M), \text{ t.q } \mathcal{d}_A(N-1) = \mathcal{d}_B(M-1)
$$

$$
\text{ Their scalar product is defined as follows: }
$$

$$
\langle A, B \rangle = \operatorname{Tr}(A^{\top} B)
= a_{zji}b_{zij}
$$
If you are still having trouble with this notation, despite the [post by “Robot Chinwag”](https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/), I recommend looking at other resources on Einstein notation and even on einsum (PyTorch function). (Basically, here we sum over the indices $$z$$)


