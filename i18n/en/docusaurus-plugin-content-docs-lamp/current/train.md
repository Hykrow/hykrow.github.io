---
title: "Training & Inference"
sidebar_position: 9
---

We will train our model on MNIST.

## Value and grad
Here we will describe the main function that the user will use. 

```rust
pub fn value_and_grad(
    params: &[Tensor], 
    build: impl Fn (&mut Trace, &[NodeId]) -> NodeId, 

) -> (Tensor, Vec<Tensor>) {
    let mut tr = Trace::new();

    let mut param_ids = Vec::with_capacity(params.len()); 

    for p in params{
        param_ids.push(tr.param(p.clone()));
    }

    let loss_id = build(&mut tr, &param_ids);

    let loss_val = tr.get_tensor(loss_id).clone(); 

    let grads = tr.backward_param_grads(loss_id);

    (loss_val, grads)
}
```

This function takes parameters `params`, which are a (reference to a) list of `Tensor`, as well as a function `build`, which is the description of the graph (and model) creation. 
`build` takes a mutable `Trace` and a (reference to a) list of `NodeId`: the parameters. It returns an id: the loss. (the end of the graph)
`value_and_grad` returns the loss and gradients of the graph. 

## Optimizers

### Sgd
Sgd is the basic optimizer. Here is a short implementation: 
```rust
pub struct Sgd{
    pub lr: f32
}

impl Sgd{
    pub fn update(&self, params: &[Tensor], grads: &[Tensor]) -> Vec<Tensor>{
        params.iter().zip(grads.iter())
            .map(|(param, grad)|
                param + &grad.apply(|x| x*(-self.lr))
        ).collect()
    }
}
```
The goal is simply to subtract the `-lr` * their gradient for each parameter to optimize their values and thus reduce the loss.
This is basic gradient descent (https://www.youtube.com/watch?v=a5xuJLFTC7o).


## Utilisation


Let's define the model, the dataloaders, and the forward function. If you want more details about dataloaders, go to [github](https://github.com/Hykrow/engine_rs/tree/main/src/dataloader). And for the [complete train example directly](https://github.com/Hykrow/engine_rs/tree/main/example).


```rust
    let mut train = DataLoader::new(ds_train, 10_000, true,  collate_train);
    let mut test  = DataLoader::new(ds_test,  2_000, false, collate_test);



    let mut params = vec![
        Linear::init_kaiming(784, 200),
        Linear::init_kaiming(200, 50), 
        Linear::init_kaiming(50, 10)
    ].concat();
    let sgd = sgd::Sgd {lr: 0.1};
    // first forward function
    fn forward_logits(tr: &mut Trace, pids: &[NodeId], x: NodeId) -> NodeId {
        let mut cur = ParamCursor::new(pids);
        // its very important to implement it in the same order 
        let l1 = Linear::bind(&mut cur);
        let l2 = Linear::bind(&mut cur);
        let l3 = Linear::bind(&mut cur);

        let h1 = l1.apply(tr, x);
        let z1 = relu(tr, h1);
        let h2 = l2.apply(tr, z1);
        let z2 = relu(tr, h2);
        l3.apply(tr, z2) // logits
    };

```

### Training
For the training, we can just do :
```rust
for epoch in 0..10 {
    train.reset_epoch();
    for (xb, yb) in &mut train {
        let (loss, grads) = value_and_grad(&params, |tr, pids| {
            let x = tr.input(xb.clone());
            let y = tr.input(yb.clone());
            let logits = forward_logits(tr, pids, x);
            softmax_crossentropy(tr, logits, y) 
        });

    println!("loss: {}", loss.data[0]);
    params = sgd.update(&params, &grads); 
    }
    println!("epoch {epoch} ok");
}
```

### Inference

And now, for the inference:
```rust
let mut correct = 0usize;
let mut total = 0usize;
test.reset_epoch();

for (xb, yb) in &mut test {
    let mut tr = Trace::new();
    
    let x = tr.input(xb.clone());
    let pids = get_params_id(&mut tr,&params); 
    let logits = forward_logits(&mut tr, &pids, x);
    let pred = tr.get_tensor(logits).argmax_last();      // [B]
    let y_true = yb.argmax_last(); 
    
    correct += pred.iter().zip(y_true.iter()).map(|(&a, &b)| if a == b {1} else {0}).sum::<usize>(); // counts the number of equalities in the batch
    total += xb.shape[0]; //  ok because 1d
}
println!("accuracy = {:.2}%", 100.0 * correct as f32 / total as f32);
```

The function `get_params_id` is very simple: 

```rust
pub fn get_params_id(tr: &mut Trace, params: &[Tensor])-> Vec<usize>{
    let mut param_ids = Vec::with_capacity(params.len()); 

    for p in params{
        param_ids.push(tr.param(p.clone()));
    }
    param_ids
}
```