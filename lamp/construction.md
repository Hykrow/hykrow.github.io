---
title: "Construction du graphe"
sidebar_position: 7
---
Maintenant qu'on a montré le calcul de ces gradients, on peut passer à l'implémentation des différentes opérations pour construire le graphe! 

## Détails - Correspondance shape

Un détail important est que le gradient doit avoir la même shape que son parent. Par ex, le gradient par rapport à $$A$$ doit avoir la même shape que $$A$$. (Sinon c'est absure). Or ce n'est plus le cas si il y a eu un broadcast dans le gradient $$G$$ précédent. Il faut sommer sur chacun des batch broadcastés pour re obtenir la shape voulue. 
### Formalisme - exemple
On va faire l'exemple sur la multiplication, mais c'est vraiment le cas pour toutes les opérations.

Soient $A\in\mathbb R^{B\times m\times n}$, $B\in\mathbb R^{n\times p}$,
$Y\in\mathbb R^{B\times m\times p}$ avec $Y_b=A_b@B$.
Soit $L:\mathbb R^{B\times m\times p}\to\mathbb R$ et $G_b=\nabla_{Y_b}L$.

$$
\begin{aligned}
\mathrm dL
&= \sum_{b=1}^{B} \langle G_b,\ \mathrm dY_b\rangle \\
&= \sum_{b=1}^{B} \left\langle G_b,\ \mathrm dA_b\, @ B \;+\; A_b @\,\mathrm dB \right\rangle \tag{★} \\
&= \sum_{b=1}^{B} \left\langle G_b@ B^{\!\top},\ \mathrm dA_b \right\rangle
 \;+\; \sum_{b=1}^{B} \left\langle A_b^{\!\top}@ G_b,\ \mathrm dB \right\rangle \\
&= \sum_{b=1}^{B} \left\langle G_b @B^{\!\top},\ \mathrm dA_b \right\rangle
 \;+\; \left\langle \sum_{b=1}^{B} A_b^{\!\top}@ G_b,\ \mathrm dB \right\rangle.
\end{aligned}
$$

Par identification (Riesz), on obtient les gradients :
$$
\boxed{\ \nabla_{A_b} L \;=\; G_b@\,B^{\!\top}\quad\text{pour chaque }b,\qquad
       \nabla_{B} L \;=\; \sum_{b=1}^{B} A_b^{\!\top} @G_b\ }.
$$
On voit bien que $$B$$, qui a été broadcasté par l'opération de multiplicaiton pour match la shape de $$A$$, doit être sommé ensuite sur les indices du broadcast.
### Implémentation
```rust
impl Tensor{
        pub fn sum_over_broadcasted_batches(&self, origin_shape : &[usize]) -> Tensor{


        let len_diff = self.shape.len()- origin_shape.len();
        let new_shape = [vec![1; len_diff], origin_shape.to_vec()].concat();
        let n = new_shape.len();
        
        

        let mut new_data= vec![0f32; new_shape.numel()]; 
        let new_strides = Tensor::compute_strides(&new_shape);
        for lin in 0..(self.shape.numel()){
            let old_idx = Tensor::idx_from_lin(&self.shape, lin);
            let mut new_idx = Vec::new();
            for i in 0..n{
                if new_shape[i] == 1{
                    new_idx.push(0);
                }else{
                    new_idx.push(old_idx[i]);
                }
            }
            let new_lin : usize= new_idx.iter().zip(new_strides.iter()).map(|(&sa, &st)| sa*st).sum();
            new_data[new_lin]+= self.get(&old_idx); 
        }

        let t = Tensor{
            data:Arc::new(new_data), 
            shape: new_shape.clone(), 
            strides: Tensor::compute_strides(&new_shape), 
            offset: 0
        }.squeeze_first(len_diff); 
        t

    }
}
```

## Opérations
Pour construire les différentes opérations, on utilise une référence vers `Trace` afin de pourvoir ajouter le nouveau noeud issu de l'opération au graphe. 

`vjp` signifie vector jacobian product. C'est juste la fonction qui calcule le gradient tel que défini précédemment. Elle prend pour argument `g_out`, qui est le gradient de $$L$$ par rapport à $$Y = f(A, B)$$ ou $$Y = f(A)$$. `move` signifie qu'il bouge (utilise) les valeurs nécessaires à la fonction définies en haut de cette dernière. (comme ça on peut les réutiliser dans la fonction)

La fonction `vjp` renvoie aussi l'id des parents des opérations associé à chaque gradient. Cela sert à bien propager et accumuler le gradient.

### Addition
```rust
pub fn add(tr: &mut Trace, a: NodeId, b: NodeId) -> NodeId{
    let va = tr.get_tensor(a).clone();
    let vb = tr.get_tensor(b).clone(); 

    let res = &va+&vb; 
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        let ga = g_out.sum_over_broadcasted_batches(&va.shape); 
        let gb = g_out.sum_over_broadcasted_batches(&vb.shape);

        smallvec![(a, ga), (b, gb)]
    };
    tr.push(Node { value: res, parents_id: smallvec![a, b], vjp: Some(Box::new(vjp)), is_param: false })
    
}

```

### Multiplication
Dans les faits, il y a des détails à prendre en compte si on manipule des scalaires / des vecteurs. Mais je ne l'ai pas ajouté ici par soucis de claireté. 
Je vous conseille de voir le [github](https://github.com/Hykrow/engine_rs/tree/main/src/ops) pour plus de détails.
```rust

pub fn matmul(tr: &mut Trace, a_id: NodeId, b_id: NodeId) -> NodeId{
    let  a= tr.get_tensor(a_id).clone();
    let  b = tr.get_tensor(b_id).clone();

    let a_rank = a.shape.len();
    let b_rank = b.shape.len();

    let c = tensor_mul(&a, &b);
    
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{

        let ga = tensor_mul(&g_out, &b.mat_transpose()).sum_over_broadcasted_batches(&a.shape);
        let gb = tensor_mul(&a.mat_transpose(), &g_out).sum_over_broadcasted_batches(&b.shape);
        
        smallvec![(a_id, ga), (b_id, gb)]
        

    }; 

    tr.push(crate::trace::Node { value: c, parents_id: smallvec![a_id, b_id], vjp: Some(Box::new(vjp)), is_param: false })
}
```
### Fonction élément par élément 
C'est très simple à implémenter: 
```rust
pub fn apply<F>(tr: &mut Trace, a_id: NodeId, f_apply: F, f_backwards: fn(f32) -> f32) -> NodeId
    where 
    F: Fn(f32) -> f32, 

{
    let a= tr.get_tensor(a_id).clone();
    let c = a.apply(f_apply);
    
    
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        smallvec![(a_id, hadamard_mul_direct(&a.apply(f_backwards), g_out).sum_over_broadcasted_batches(&a.shape))] 

    }; 

    tr.push(crate::trace::Node { value: c, parents_id: smallvec![a_id], vjp: Some(Box::new(vjp)), is_param: false })
}
```
Elle prend en argument la fonction `f_apply` et sa dérivée `f_backwards`
#### Tanh, ReLU
Ca devient très simple:
```rust

pub fn tanh(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.tanh(), |x| 1f32-x.tanh()*x.tanh())
}

pub fn relu(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.max(0f32), |x| if x >= 0f32 {1f32} else{0f32})
}
```
## Backward pass

Maintenant qu'on a bien construit le graphe et que chaque noeud à sa `vjp` (ou non), on a plus qu'a appeler ces fonctions dans le bon ordre. 

### Récupérer les gradients des parents
```rust
impl Trace{
    pub fn accum(slot: &mut Option<Tensor>, delta: Tensor)
    {
        *slot =Some(
            match slot.take(){
                Some(g) => &g+&delta,
                None =>delta
            }
        )
    }

    pub fn backward_param_grads(&self, root: NodeId) -> Vec<Tensor>
    {
        let order = self.order(root);

        let mut grads: Vec<Option<Tensor>> = vec![None; self.len()];
        grads[root] = Some(Tensor::ones(&self.get_tensor(root).shape));

        for &node_id in order.iter(){
            let Some(ref g_out) = grads[node_id] else {continue}; // permet de chopper le g_out dans le Some directement. Normalement déjà dedans car gradient déjà calculé par l'ordre topologique 
            if let Some(ref vjp) = self.nodes[node_id].vjp{ // si il a une vector jacobian product
                for (parent_id, tensor) in vjp(g_out){
                    Trace::accum(&mut grads[parent_id], tensor);
                }
            }
        }

        let mut res = Vec::with_capacity(self.params_id.len());
        for &id in &self.params_id{
            res.push(grads[id].as_ref().unwrap_or_else(|| panic!("dynamic: le gradient d'un param n'est pas trouvable")).clone());
        }
        res

    }
}
```

On a donc une méthode pour obtenir les gradients du graphe des opérations.