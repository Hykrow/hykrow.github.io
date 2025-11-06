---
title: "Entraînement & Inférence"
sidebar_position: 9
---

On va entraîner notre modèle sur MNIST. 

## Value and grad
On va décrire ici la fonction principale que l'utilisateur va utiliser. 
```rust
pub fn value_and_grad(
    params: &[Tensor], 
    build: impl Fn (&mut Trace, &[NodeId]) -> NodeId, 

) -> (Tensor, Vec<Tensor>) {
    let mut tr = Trace::new();

    let mut param_ids = Vec::with_capacity(params.len()); 

    // ca permet d'avoir leur id. Ca nous serira pour re ecrire 
    for p in params{
        param_ids.push(tr.param(p.clone()));
    }

    let loss_id = build(&mut tr, &param_ids);

    let loss_val = tr.get_tensor(loss_id).clone(); 

    let grads = tr.backward_param_grads(loss_id);

    (loss_val, grads)
}
```

Cette fonction prend en argument des paramètres `params` qui sont une (référence vers une) liste de `Tensor`, ainsi qu'une fonction `build` qui est la description de la création du graphe (et du modèle). 
`build` prend en argument une `Trace` mutable et une  (référence vers une) liste de `NodeId`: les paramètres. Elle renvoie un id: la loss. (la fin du graphe)
`value_and_grad` renvoie la loss et les gradients du graphe. 

## Optimizers

### Sgd
Sgd est l'optimiseur de base, voici une implémentation courte: 
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
Le but est juste de soustraire au paramètres `-lr` de leur gradient pour optimiser leur valeurs et donc réduire la loss. 
C'est la [descente de gradient](https://www.youtube.com/watch?v=a5xuJLFTC7o) basique.


## Utilisation


Définissons le modèle, les dataloaders ainsi que la fonction forward. Si vous voulez plus de détails sur les dataloaders, allez vous le [github](https://github.com/Hykrow/engine_rs/tree/main/src/dataloader). Et pour l'[exemple de train complet directement](https://github.com/Hykrow/engine_rs/tree/main/example)

```rust
    let mut train = DataLoader::new(ds_train, 10_000, true,  collate_train);
    let mut test  = DataLoader::new(ds_test,  2_000, false, collate_test);



    let mut params = vec![
        Linear::init_kaiming(784, 200),
        Linear::init_kaiming(200, 50), 
        Linear::init_kaiming(50, 10)
    ].concat();
    let sgd = sgd::Sgd {lr: 0.1};

    // on définit une première fonction pour le forward. Comme la fonction build mais prend en plus l'id de l'input
    fn forward_logits(tr: &mut Trace, pids: &[NodeId], x: NodeId) -> NodeId {
        let mut cur = ParamCursor::new(pids);
        // IL EST TRES IMPORTANT DE BIND DANS LE MEME ORDRE QUE PREVU 
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

### Entrainement
Pour l'entrainement, on peut juste faire :

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

Et maintenant, pour le test: 
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
    
    correct += pred.iter().zip(y_true.iter()).map(|(&a, &b)| if a == b {1} else {0}).sum::<usize>(); // compte les égaux dans le batch
    total += xb.shape[0]; // ici ok car 1d de batch et 1d de vecteur
}
println!("accuracy = {:.2}%", 100.0 * correct as f32 / total as f32);
```

La fonction `get_params_id` est très simple: 

```rust
pub fn get_params_id(tr: &mut Trace, params: &[Tensor])-> Vec<usize>{
    let mut param_ids = Vec::with_capacity(params.len()); 

    for p in params{
        param_ids.push(tr.param(p.clone()));
    }
    param_ids
}
```