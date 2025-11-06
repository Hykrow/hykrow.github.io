---
title: "Utils, linear layers, inits & losses"
sidebar_position: 8
---
In this function, I just detail the utilities needed to do the training (coming soon!).

## Linear Layer

### Implementation
With everything we've coded, it becomes very easy to make this layer! 
```rust
pub struct Linear{
    pub w: NodeId,
    pub b: NodeId,
}

impl Linear{
    pub fn bind(cur: &mut ParamCursor) -> Linear{
        let (w, b) = cur.take2(); 
        Linear{w, b}
    }

    // x.w + b
    pub fn apply(&self, tr: &mut Trace, x: NodeId) -> NodeId{
        let x_dot_w = matmul(tr, x, self.w); 
        add(tr, x_dot_w, self.b)
    }

    pub fn init_kaiming(in_dim: usize, out_dim: usize)-> Vec<Tensor>{
        vec![kaiming(in_dim, out_dim), Tensor::zeros(&[out_dim])]

    }
}
```
### Utilitaires
You may have noticed the `bind` function in `Linear`: 
This is to make it easier to call this layer. We define: 

```rust
pub struct ParamCursor<'a>{
    p: &'a [NodeId], 
    i: usize, 
}

impl <'a>ParamCursor<'a>{
    pub fn new (p: &'a [NodeId]) -> Self{Self {p, i:0}}

    pub fn take(&mut self) -> NodeId{
        let id = self.p[self.i]; 
        self.i+=1; 
        id
    }

    pub fn take2(&mut self) -> (NodeId, NodeId){
        (self.take(), self.take())
    }

    pub fn remaining(&self) -> usize{
        self.p.len() - self.i
    }

}
```
This will allow us to avoid manipulating the `NodeId` identifiers of the parameters directly. If you are having trouble understanding its usefulness, skip this and come back to it after reading the training section (the next one).

## Initialization

For this part, I highly recommend this [video](https://www.youtube.com/watch?v=eoNVmZDnn9w) and this [video](https://www.youtube.com/watch?v=1Rf7BVQ-z0M). Watch both of them.

The code is as follows: 

```rust
pub fn kaiming(in_dim: usize, out_dim: usize) -> Tensor{
    let limit = f32::sqrt(6f32/(in_dim as f32));
    let mut rng = rand::thread_rng();
    let mut vec = Vec::with_capacity(in_dim*out_dim);
    for _ in 0..in_dim*out_dim{
        vec.push(rng.gen_range((-limit)..(limit)));
    }
    Tensor::from_vec(&vec, &[in_dim, out_dim]).unwrap()
}
```

## Loss functions

First, we need an operation that calculates the average loss across batches. This is because, in the case of batch training/inference, we may end up with a tensor rather than a simple scalar.
```rust
pub fn mean_all(tr: &mut Trace, x_id: NodeId) -> NodeId{
    let x = tr.get_tensor(x_id).clone(); 
    let n = x.shape.numel();
    let y = x.sum_all(); 


    let vjp = move |g_out: &Tensor|  -> SmallVec<[(NodeId, Tensor); 2]>{
        let gx = g_out.apply(|x| x/(n as f32)).broadcast_view(&x.shape).unwrap();
        smallvec![(x_id, gx)]
    };
    tr.push(crate::trace::Node { value: y, parents_id: smallvec![x_id], vjp: Some(Box::new(vjp)), is_param: false })
}
```

### MSE
We recall the MSE here: 
$$
\mathrm{MSE}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n} \|\mathbf{y} - \hat{\mathbf{y}}\|_2^2
$$

For the mse, I just used the different operations already used before, as well as others created for this loss.
This way, there is no need to calculate the gradient itself, since we just take the squared difference. All you have to do is use `funcitons::apply`, give it the function to apply element by element and its derivative. 

```rust
pub fn mse(tr: &mut Trace, pred_id: NodeId, target_id: NodeId) -> NodeId{ 

    let diff_id = sub(tr, pred_id, target_id);

    let square_id = functions::apply(tr, diff_id, |x| x*x, |x| 2f32*x);
    
    mean_all(tr, square_id)
}

```



### Softmax cross-entropy
For softmax cross-entropy, I highly recommend this [video](https://www.youtube.com/watch?v=RQ0zVyicqxg) and this one [here](https://www.youtube.com/watch?v=rf4WF-5y8uY) (by the same author as the video on initialization).


```rust
pub fn softmax(t: &Tensor) -> (Tensor, Tensor){
    let n = t.shape.len();
    assert!(t.shape[n-1] > 0);
    
    // we need the unsqueeze so that the broadcast operates correctly
    let m = t.max_last();


    let scaled = t- &m;

    let exp = scaled.apply(f32::exp);

    let s = exp.sum_last();

    let lse = &m+ &s.apply(f32::ln);
    let softmax = &exp/ &s; 
    (lse, softmax)

}
pub fn softmax_crossentropy(tr: &mut Trace, logits_id: NodeId, target_id: NodeId) -> NodeId{
    let logits = tr.get_tensor(logits_id);
    let y = tr.get_tensor(target_id);
    let (lse, softmaxed) = softmax(logits);

    // multiplication element apr element => sum last => moyenne pondérée du label voulu predit
    let zy = (logits*y).sum_last();

    let value = &lse - &zy; 


    let soft_c = softmaxed.clone();
    let y_c = y.clone();
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{

        let diff = &soft_c - &y_c;
        smallvec![(logits_id, &diff*g_out)]
    };
    let smxcpy = tr.push(Node { value: value, parents_id: smallvec![logits_id], vjp: Some(Box::new(vjp)), is_param: false });
    
    mean_all(tr, smxcpy)
}
```


## Appendix

If you want to see the different functions `max_last`, `sum_last`, `sub`, etc., here they are: 
```rust
impl Tensor{
    // suppose that keepdim = true;
    pub fn apply_and_reduce_last(&self, f: fn(f32, f32) -> f32, neutral_el: f32) -> Tensor{
        assert!(self.shape.len() >= 1);
        
        let n = self.shape.len();
        let batches_shape= &self.shape[0..(n-1)];
        let batches_number = batches_shape.numel();
        let mut batches = vec![neutral_el; batches_number];

        assert!(self.shape[n-1] > 0);
        for lin in 0..self.shape.numel(){
            let x = self.get_from_lin(lin);
            batches[lin/self.shape[n-1]] = f(batches[lin/self.shape[n-1]], x);
        }
        Tensor::from_vec(&batches, batches_shape).unwrap().unsqueeze_view(n-1)

    }
    // suppose that keepdim = true;
    pub fn sum_last(&self) -> Tensor{
        self.apply_and_reduce_last(|x, y| x+y, 0f32)
    }
    // suppose that keepdim = true;
    pub fn max_last(&self) -> Tensor{
        self.apply_and_reduce_last(|x, y| x.max(y), f32::NEG_INFINITY)
    }

    //suppose that keepdim = true; 
    pub fn argmax_last(&self)-> Vec<usize>{
        assert!(self.shape.len() >= 1);
        
        let n = self.shape.len();
        let batches_shape= &self.shape[0..(n-1)];
        let batches_number = batches_shape.numel();
        let mut batches = vec![f32::NEG_INFINITY; batches_number];
        let mut args = vec![0; batches_number];

        for lin in 0..self.shape.numel(){
            let x = self.get_from_lin(lin);
            let new_idx = lin/self.shape[n-1];
            if batches[new_idx] < x{
                batches[new_idx] = x; 
                args[new_idx] = lin%self.shape[n-1];
            }
        }

        args


    }
}
pub fn sub(tr: &mut Trace, a: NodeId, b: NodeId) -> NodeId{ 
    let va = tr.get_tensor(a).clone();
    let vb = tr.get_tensor(b).clone(); 

    let res = &va-&vb; 
    let vjp = move |g_out: &Tensor| -> SmallVec<[(NodeId, Tensor); 2]>{
        let ga = g_out.sum_over_broadcasted_batches(&va.shape); 
        let gb = g_out.sum_over_broadcasted_batches(&vb.shape).apply(|x| x*(-1f32));

        smallvec![(a, ga), (b, gb)]
    };
    tr.push(Node { value: res, parents_id: smallvec![a, b], vjp: Some(Box::new(vjp)), is_param: false })
}
impl Sub for &Tensor{
    type Output = Tensor;
    fn sub(self, b: &Tensor) -> Tensor{

        self+ &b.apply(|x| x*-1f32)
    }
}
   
```