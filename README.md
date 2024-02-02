## Bayesian Spatiotemporal Point Process

This package provides bayesian inference for three spatiotemporal point process models with or without spatial covariates:
- Log Gaussian Cox Process (lgcp)
- Hawkes Process
- Cox Hawkes Process


### Usage

Install with
```pip install git+https://github.com/imanring/BSTPP.git```

API documentation is in ```bstpp_API_doc.pdf```.

See ```demo.ipynb``` for a demo.

### Model Details

The full Cox Hawkes Model is formulated as follows,


$\lambda(t,s) = \mu(t,s) + \sum_{i:t_i < t}{\alpha f(t-t_i;\beta) \varphi(s-s_i;\sigma)}$



$f$ by default is the exponential density and $\varphi$ by default is the normal density.



$\mu(t,s) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$



$X(s)$ is the spatial covariate matrix, and $f_s$ and $f_t$ are gaussian processes.


The Hawkes process is the same with the as the Cox Hawkes, except the background is


$\mu(t,s) = exp(a_0 + X(s)w)$


Finally, the Log Gaussian Cox Process is the same as Cox Hawkes except without the self-exciting summation,


$\lambda(t,s) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$

### Acknowledgements

This repo is based on code from [1]. The trained decoders and encoder/decoder functions are provided by Dr Elisaveta Semenova following the proposals in [2]. 


[1] X. Miscouridou, G. Mohler, S. Bhatt, S. Flaxman, S. Mishra, Cox-Hawkes: Doubly stochastic spatiotemporal poisson point process, Transaction of Machine Learning Research, 2023

[2] Elizaveta Semenova, Yidan Xu, Adam Howes, Theo Rashid, Samir Bhatt, B. Swapnil Mishra, and Seth R.
Flaxman. Priorvae: encoding spatial priors with variational autoencoders for small-area estimation. Royal
Society Publishing, pp. 73–80, 2022 

