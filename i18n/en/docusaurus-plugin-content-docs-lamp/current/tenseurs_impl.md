---
title: "Implementation Details - Tensors"
sidebar_position: 3

---

Here, I will detail the implementation of tensors. You can find the code [here](https://github.com/Hykrow/engine_rs/tree/main/src/tensor)

## Tensors


### Representation
Here is the struct that represents tensors: 

```rust
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Arc<Vec<f32>>,
    pub shape : Vec<usize>, 
    pub strides : Vec<usize>, 
    pub offset : usize,
}
```
`Arc` means a reference counter that is compatible with multiple threads. 
We need this because we are going to copy a lot of tensors, and we cannot copy all the data again. (And there is no GC in Rust.)

`data` is the 1D vector of data, `shape` is the $shape$ of the tensor ($$A \in \mathbb{R}^{shape} $$)

`strides` is the data indicating how much we need to “skip” for each index in the 1D representation to obtain an N-D representation.

#### Strides, Shapes

Here is how strides are implemented: 

```rust
impl Tensor{

    pub fn compute_strides(shape : &[usize]) -> Vec<usize>{
        let n = shape.len();
        let mut strides = vec![0; n];
        let mut product = 1; 
        for (i, dim) in shape.iter().rev().enumerate(){
            strides[n-i-1] = product; 
            product*=dim;
        }
        strides
    }
}
```
And to get an element : 
```rust
impl Tensor{
    pub fn get(&self, id: &[usize]) -> f32{   
        let idx = id.iter().zip(self.strides.iter()).map(|(&a, &b)| a*b).sum::<usize>();
        self.data[idx+self.offset]
    }
}
```

We can also have a linear representation of the indices (1D). This is useful for iterating over them later: 

```rust
impl Tensor{

    pub fn idx_from_lin(shape: &[usize], mut lin: usize) -> Vec<usize>{

        let mut idxs: Vec<usize> = shape.iter().rev().map(|&sa|
            if sa!=0 {
                let idx = lin%sa;
                lin/=sa;
                idx
            }else{
                0
            }
        ).collect();

        // be careful of this ! +
        idxs.reverse();
        idxs
    }
    pub fn lin_from_idx(&self, id: &[usize])-> usize{
        let idx = id.iter().zip(self.strides.iter()).map(|(&a, &b)| a*b).sum::<usize>();
        idx
    }
}
```
#### Exemple
Example of use: 
```rust

//tensor représentation : [ [1, 2], [3, 4] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 

let b: f32 = a.get(&[0, 0, 1]); // 2f32.
let c: f32 = a.get(&[0, 1, 0]); // 3f32.

```
### Squeeze, Unsqueeze


Something useful for our operations are squeeze and unsqueeze. They are used to remove or add virtual dimensions. For example, in the previous example, we could have squeezed dimension 0 because it served no purpose. 


```rust
impl Tensor{
        pub fn unsqueeze_view(&self, axis: usize) -> Tensor {
        let r = self.shape.len();
        assert!(axis <= r, "unsqueeze axis hors limites");

        let mut shape = self.shape.clone();
        shape.insert(axis, 1);

        let mut strides = self.strides.clone();
        let inserted = if axis == r {
            1
        } else {
            self.strides[axis] * self.shape[axis]
        };
        strides.insert(axis, inserted);

        Tensor {
            data: self.data.clone(),
            shape,
            strides,
            offset: self.offset,
        }
    }

    pub fn squeeze_view(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(), "squeeze axis out of bounds");
        assert!(self.shape[axis] == 1, "squeeze: the dimension is not 1");

        let mut shape = self.shape.clone();
        shape.remove(axis);
        let mut strides = self.strides.clone();
        strides.remove(axis);

        Tensor {
            data: self.data.clone(),
            shape,
            strides,
            offset: self.offset,
        }
    }
}
```

#### Exemple
Use example: 
```rust

// tensor représentation :  : [ [1, 2], [3, 4] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 


let b: Tensor = a.squeeze_view(0); // [1, 2], [3, 4]. 
                                   // shape: [2, 2]
```

### Broadcast

```rust
impl Tensor{
    pub fn broadcast_view(&self, a: &[usize]) -> Result<Tensor, String>{
        //left pad d'abord: 
        if a.len() < self.shape.len(){
            return Err("len de la shape de la cible trop petite".into());
        }
        
        let mut res: Tensor = self.unsqueeze_first(a.len()-self.shape.len()); // assumes that the shape we want had a size >= at the one we will have
        assert_eq!(res.shape.len(), a.len(), "enorme bug broadcast_view");
        //recalculate strides
        let n  = a.len();
        for i in 0..n{
            if res.shape[i] < a[i] && res.shape[i] ==1 {
                res.shape[i] =  a[i];
                res.strides[i] = 0;
            }else if res.shape[i] != a[i] && res.shape[i] != 1 { // not the same shape mais pas broadcastable...
                return Err("broadcast_view : non broadcastable..".into());
            }
        }
       Ok(res)
        
    }
    // gives the needed shape for the two tensors
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String>{
        let n = a.len().max(b.len());

        let mut ita = a.iter().rev().copied().chain(iter::repeat(1)); 
        let mut itb = b.iter().rev().copied().chain(iter::repeat(1));

        let mut res = Vec::with_capacity(n);

        for _ in 0..n{ // juste to iterer n times on iterators..
      
            let el_a = ita.next().unwrap();
            let el_b = itb.next().unwrap();
 
            if el_a == 1 || el_b == 1 || el_a == el_b { // we dupe the scalar( the one whos dimension is 1)
                res.push(el_a.max(el_b));
            }else{
                return Err("Error broadcast_shape".into());
            }
        }
        res.reverse();
        Ok(res)

    }
}
```

#### Example

```rust
// tensor représentation : [ [1, 2], [3, 4] ]
let a: Tensor = Tensor{data: vec![1f32, 2f32, 3f32, 4f32],
               shape: vec![1, 2, 2], 
               strides: compute_strides(&[1, 2, 2]), 
               offset: 0}; 


// tensor représentation :[ [1, 2], [3, 4] ], [ [1, 2], [3, 4] ]
let b: Tensor = a.broadcast_view(&[2, 2, 2]);


```

### Transpose
```rust
impl Tensor{
        pub fn mat_transpose(&self) -> Tensor{
        let dim = self.shape.len();
        if dim == 1{
            self.clone()
        }else{
            let mut new_shape = self.shape.clone(); 
            let mut new_strides = self.strides.clone();
            new_shape.swap(dim-1, dim-2);
            new_strides.swap(dim-1, dim-2);
            Tensor { data: self.data.clone(), shape: new_shape.clone(), strides: new_strides, offset: self.offset }
        }

    }


}

```
