---
title: "Graph construction"
sidebar_position: 7
---
Now that we have shown how to calculate these gradients, we can move on to implementing the various operations to construct the graph!

## Details - Shape correspondence


An important detail is that the gradient must have the same shape as its parent. For example, the gradient with respect to $$A$$ must have the same shape as $$A$$. (Otherwise, it is absurd). However, this is no longer the case if there has been a broadcast in the previous gradient $$G$$. We must sum over each of the broadcasted batches to obtain the desired shape. 
### Formalism - example
We will use multiplication as an example, but this is true for all operations.


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

By identification (Riesz), we obtain the gradients:
$$
\boxed{\ \nabla_{A_b} L \;=\; G_b@\,B^{\!\top}\quad\text{pour chaque }b,\qquad
       \nabla_{B} L \;=\; \sum_{b=1}^{B} A_b^{\!\top} @G_b\ }.
$$
It is clear that $$B$$, which was broadcast by the multiplication operation to match the shape of $$A$$, must then be summed over the indices of the broadcast.
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

## Operations

To construct the various operations, we use a reference to `Trace` so that we can add the new node resulting from the operation to the graph. 

`vjp` stands for vector jacobian product. It is simply the function that calculates the gradient as defined above. It takes `g_out` as an argument, which is the gradient of $$L$$ with respect to $$Y = f(A, B)$$ or $$Y = f(A)$$. `move` means that it moves (uses) the values needed for the function defined at the top of the function. (so that we can reuse them in the function)

The `vjp` function also returns the ID of the parent operations associated with each gradient. This is used to properly propagate and accumulate the gradient.



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
In practice, there are details to consider when manipulating scalars/vectors. But I haven't included them here for the sake of clarity.
I recommend checking out [github](https://github.com/Hykrow/engine_rs/tree/main/src/ops) for more details.


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
### Element-by-element function
It's very simple to implement: 
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
It takes the function `f_apply` and its derivative `f_backwards` as arguments.
#### Tanh, ReLU
It becomes very simple:
```rust

pub fn tanh(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.tanh(), |x| 1f32-x.tanh()*x.tanh())
}

pub fn relu(tr: &mut Trace, a_id: NodeId) -> NodeId{
    apply(tr, a_id, |x| x.max(0f32), |x| if x >= 0f32 {1f32} else{0f32})
}
```
## Backward pass

Now that we have constructed the graph and each node has its `vjp` (or not), we just need to call these functions in the right order. 

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
            let Some(ref g_out) = grads[node_id] else {continue}; // easier way to put a match
            if let Some(ref vjp) = self.nodes[node_id].vjp{ // if he has a vjp
                for (parent_id, tensor) in vjp(g_out){
                    Trace::accum(&mut grads[parent_id], tensor);
                }
            }
        }

        let mut res = Vec::with_capacity(self.params_id.len());
        for &id in &self.params_id{
            res.push(grads[id].as_ref().unwrap_or_else(|| panic!("dynamic: le gradient of a parameter is not found")).clone());
        }
        res

    }
}
```

We therefore have a method for obtaining the gradients of the operation graph.