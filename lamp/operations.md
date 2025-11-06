---
title: "Détails d'implémentation - opérations"
sidebar_position: 4
---

Ici on détaille les opérations de base. Aussi dispo en plus détaillé [ici](https://github.com/Hykrow/engine_rs/tree/main/src/ops)


## Addition
```rust

impl Add for &Tensor{
    type Output = Tensor;
    fn add(self, b: &Tensor) -> Tensor{

        let broadcast_shape = Tensor::broadcast_shape(&self.shape, &b.shape).unwrap();

        
        
        let sz = broadcast_shape.numel();
        let mut vec = Vec::with_capacity(sz);       
        let a_b = self.broadcast_view(&broadcast_shape).unwrap();
        let b_b = b.broadcast_view(&broadcast_shape).unwrap();


        for lin in 0..sz{
            vec.push(a_b.get_from_lin(lin)+ b_b.get_from_lin(lin));
            
        }
        Tensor { data: Arc::new(vec), shape: broadcast_shape, strides: a_b.strides, offset: 0 }
    }
}

```
### Exemple
```rust
// représentation du tenseur : [ [ [1, 2], [3, 4] ] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 


// représentation du tenseur:[ [[1], [1]] ]
let b: Tensor = Tensor{data: vec![1f32, 1f32],
               shape: vec![1, 1, 1], 
               strides: compute_strides(&[1, 1, 1]), 
               offset: 0}; 

let c = &a + &b; //[[2, 3], [4, 5]]

```


## Multiplication hadamard (élément par élément)

```rust

pub fn hadamard_mul_direct(a: &Tensor, b: &Tensor ) -> Tensor{

    let out_shape = Tensor::broadcast_shape(&a.shape, &b.shape).unwrap();

    let a_broad = a.broadcast_view(&out_shape).unwrap();
    let b_broad = b.broadcast_view(&out_shape).unwrap();

    let mut c = Vec::with_capacity(out_shape.numel());
    for lin in 0..out_shape.numel(){
        c.push(a_broad.get_from_lin(lin)*b_broad.get_from_lin(lin));
    }
    
    Tensor::new(Arc::new(c), &out_shape, 0)
}
```
### Exemple

```rust
// représentation du tenseur : [ [ [1, 2], [3, 4] ] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 


// représentation du tenseur:[ [[2], [1]] ]
let b: Tensor = Tensor{data: vec![2f32, 1f32],
               shape: vec![1, 1, 1], 
               strides: compute_strides(&[1, 1, 1]), 
               offset: 0}; 

let c = hadamard_mul_direct(&a, &b); //[ [[2, 4], [3, 4]] ]

```

## Multiplication matricielle par batch (matmul sur les 2 dernieres batch)

C'est ce qu'on entend par `A@B` en PyTorch.

### MatMul 2D
On va commencer par la multiplication de matrices 2d, cela va nous aider pour plus tard
```rust

impl Tensor{
    pub fn vue2d(&self, offset: usize) -> Tensor{
        let dim = self.shape.len();
        Tensor { data: self.data.clone(), shape: self.shape[(dim-2)..dim].to_vec(), strides: self.strides[(dim-2)..dim].to_vec(), offset: offset }
    }
}

// fonction helper
fn matmul2d_vec(a: &Tensor, b: &Tensor) -> Vec<f32>{
    assert_eq!(a.shape.len(), 2, "matmul2d: A doit être 2D, shape={:?}", a.shape);
    assert_eq!(b.shape.len(), 2, "matmul2d: B doit être 2D, shape={:?}", b.shape);

    let (m, p) = (a.shape[0], a.shape[1]); // A: (m, n)
    let (p2, n) = (b.shape[0], b.shape[1]); // B: (n, p)
    assert_eq!(p, p2, "matmul2d: dimensions incompatibles");



    let mut c = Vec::with_capacity(m*n);
    for i in 0..m{
        for j in 0..n{
            let mut sum = 0.0;
            for k in 0..p{
                sum+=a.get2(i, k)*b.get2(k, j); 
            }
            c.push(sum);
        }
    }
    c
}
```

### Cas Tensoriel
Enfin, passons au cas général: 
```rust
fn tensor_mul(a : &Tensor, b: &Tensor) -> Tensor{

    let a_order = a.shape.len(); 
    let b_order = b.shape.len();

    assert_eq!(a.shape[a_order-1], b.shape[b_order-2], "dimensions non ok; affichage des tenseurs : Tenseur a: {} \n Tenseur b: {} ", a, b);

    let m = a.shape[a_order-2];
    let n = b.shape[b_order-1];
    
    let p = a.shape[a_order-1];

    let batch_a = &a.shape[0..a_order-2];
    let batch_b = &b.shape[0..b_order-2]; 



    let batch = Tensor::broadcast_shape(&batch_a, &batch_b).unwrap();

    let mut out_shape = batch.clone(); 
    out_shape.push(m); 
    out_shape.push(n);

    let a_b = a.broadcast_view(&[batch.clone(), vec![m, p]].concat()).unwrap();
    let b_b = b.broadcast_view(&[batch.clone(), vec![p, n]].concat()).unwrap();
    
    let mut c = Vec::with_capacity(out_shape.numel());

    let lin_max = batch.numel();
    for lin in 0..lin_max{ // iterer a travers les batch
        let idx_from_lin = Tensor::idx_from_lin(&batch, lin);
        let a_b_2d = a_b.vue2d(a_b.batch_offset(&idx_from_lin));
        let b_b_2d = b_b.vue2d(b_b.batch_offset(&idx_from_lin));

        c.extend(matmul2d_vec(&a_b_2d, &b_b_2d));
 
    }
    Tensor::from_vec(&c, &out_shape).unwrap()

}
```
### Exemple
```rust
// représentation du tenseur : [ [ [1, 2], [3, 4] ] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 


// représentation du tenseur: [ [[1, 2], [3 ,4]], [[5, 6], [7, 8]]  ]
let b: Tensor = Tensor{data: vec![2f32, 1f32],
               shape: vec![2, 2, 2], 
               strides: compute_strides(&[2, 2, 2]), 
               offset: 0}; 


/*
[
  [[7, 10],
   [15, 22]],

  [[19, 22],
   [43, 50]]
]
*/
let c = tensor_mul(&a, &b); 

```

Bien sur, cela ne marchera pas pour des scalaires, ou avec des vecteurs; regardez le github pour ces détails