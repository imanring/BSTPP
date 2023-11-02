## Bayesian Spatiotemporal Point Process

This package provides bayesian inference for three spatiotemporal point process models:
- Log Gaussian Cox Process (lgcp)
- Hawkes Process
- Cox Hawkes Process


Install with
```pip install git+https://github.com/imanring/Cox_Hawkes_Cov.git```


See ```demo.ipynb``` for a demo.


This code allows you to do inference on self-exciting point processes with inhomogeneous backgrounds and spatial covariates. It is based on code from [1]. The trained decoders and encoder/decoder functions are provided by Dr Elisaveta Semenova following the proposals in [2]. 


[1] X. Miscouridou, G. Mohler, S. Bhatt, S. Flaxman, S. Mishra, Cox-Hawkes: Doubly stochastic spatiotemporal poisson point process, Transaction of Machine Learning Research, 2023

[2] Elizaveta Semenova, Yidan Xu, Adam Howes, Theo Rashid, Samir Bhatt, B. Swapnil Mishra, and Seth R.
Flaxman. Priorvae: encoding spatial priors with variational autoencoders for small-area estimation. Royal
Society Publishing, pp. 73–80, 2022 

