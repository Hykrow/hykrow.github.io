---
id: intro
title: "Introduction to Automatic Differentiation."
sidebar_position: 1
---

Hello!

I'll walk you through how I implemented an autodiff engine from scratch.
If you’d like, here’s the [*GitHub repo*](https://github.com/Hykrow/engine_rs).

This project originally stems from [Andrej Karpathy’s video](https://www.youtube.com/watch?v=VMj-3S1tku0) on the topic. But I noticed that when generalizing to tensors, there were very few truly **comprehensive** resources that explained everything the way I wanted. The goal here is to reference what already exists and write the missing details to arrive at something genuinely complete.

This tutorial goes deep into the mathematical definitions as well as the code. It assumes at least a second-year undergraduate level in mathematics (or equivalent) and a minimal idea of what such an engine does (think PyTorch, TensorFlow, JAX).

I chose Rust because strongly typed languages force you to better understand what you write, and since Rust is meticulous about memory, it pushes me to understand my code even more for it to work.
