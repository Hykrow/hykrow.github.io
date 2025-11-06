---
title: "Conclusion"
sidebar_position: 10
---
I trained the model for 20 minutes (5 epochs, with 6,000 training and 600 test) and obtained an accuracy of around 60-70%. The problem with the code is that it is very simple in order to be readable, but this makes it very slow. The operations need to be parallelized on the CPU (which is possible because Arc allows it), or even multiplied using the GPU. Such low accuracy can be explained by the fact that I only trained on 10% of the dataset... (and with not many epochs).

## Areas for improvement & other considerations

I therefore plan to implement CPU parallelization, as well as other layers such as regularization, convolution, etc. Perhaps even multiplication on the GPU if I have the time, but this post will not necessarily be updated. So feel free to check out [GitHub](https://github.com/Hykrow/engine_rs) if you want.

Feel free to star the repo if you like it/liked it! And contribute (to the site or the repo directly) if you want!