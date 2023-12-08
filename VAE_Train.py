# general libraries
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.example_libraries import stax

from functools import partial

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive
from numpyro.diagnostics import hpdi

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--length_scale", help="Inverse Gamma Scale Parameter for GP length",type=float,default=15)
parser.add_argument("--length_shape", help="Inverse Gamma Scale Parameter for GP length",type=float,default=1)
parser.add_argument("--var_scale", help="Log Normal Scale Parameter for GP variance",type=float,default=.5)
parser.add_argument("--var_loc", help="Log Normal Location Parameter for GP variance",type=float,default=0)
parser.add_argument("--hidden_dim1", help="Hidden Dimension Layer 1",type=int,default=45)
parser.add_argument("--hidden_dim2", help="Hidden Dimension Layer 2",type=int,default=30)
parser.add_argument("--z_dim", help="Hidden Dimension Layer 1",type=int,default=15)
parser.add_argument("--num_epochs", help="Number of Epochs of Training",type=int,default=50)
uargs = parser.parse_args()


def dist_euclid(x, z):
    x = jnp.array(x) 
    z = jnp.array(z)
    if len(x.shape)==1:
        x = x.reshape(x.shape[0], 1)
    if len(z.shape)==1:
        z = x.reshape(x.shape[0], 1)
    n_x, m = x.shape
    n_z, m_z = z.shape
    assert m == m_z
    delta = jnp.zeros((n_x,n_z))
    for d in jnp.arange(m):
        x_d = x[:,d]
        z_d = z[:,d]
        delta += (x_d[:,jnp.newaxis] - z_d)**2
    return jnp.sqrt(delta)


def exp_sq_kernel(x, z, var, length, noise=0, jitter=1.0e-6):
    #dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist_/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k


def GP(gp_kernel, x, jitter=1e-5, var=None, length=None, y=None, noise=False):
    
    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(uargs.length_scale,uargs.length_shape))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(uargs.var_loc,uargs.var_scale))
        
    k = gp_kernel(x, x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.1))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

args = {"n": 625,
        "gp_kernel": exp_sq_kernel,
        "rng_key": random.PRNGKey(2),
        "batch_size": 10
}

n_xy = 25
grid = jnp.arange(0, 1, 1/n_xy)
u, v = jnp.meshgrid(grid, grid)
x_xy = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
dist_ = dist_euclid(x_xy, x_xy)
rng_key, rng_key_predict = random.split(random.PRNGKey(4))
gp_predictive = Predictive(GP, num_samples=args["batch_size"])


def vae_encoder(hidden_dim1, hidden_dim2, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Elu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()), # mean
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
        ),
    )


def vae_decoder(hidden_dim1, hidden_dim2, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim1, W_init=stax.randn()),
        stax.Elu,
        stax.Dense(hidden_dim2, W_init=stax.randn()),
        stax.Elu,
        stax.Dense(out_dim, W_init=stax.randn()) 
    )

def vae_model(batch, hidden_dim1, hidden_dim2, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", vae_decoder(hidden_dim1, hidden_dim2, out_dim), (batch_dim, z_dim))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((batch.shape[0],z_dim)), jnp.ones((batch.shape[0],z_dim))))
    gen_loc = decode(z)    
    return numpyro.sample("obs", dist.Normal(gen_loc, .1), obs=batch) 
    

def vae_guide(batch, hidden_dim1, hidden_dim2, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", vae_encoder(hidden_dim1, hidden_dim2, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    z = numpyro.sample("z", dist.Normal(z_loc, z_std))
    return z


@jit
def epoch_train(rng_key, svi_state, num_train):

    def body_fn(i, val):
        rng_key_i = random.fold_in(rng_key, i) 
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        loss_sum, svi_state = val # val -- svi_state
        batch = gp_predictive(rng_key_i, gp_kernel=args["gp_kernel"], x=args["x"], jitter=1e-4)
        svi_state, loss = svi.update(svi_state, batch['y']) 
        loss_sum += loss / args['batch_size']
        return loss_sum, svi_state

    return lax.fori_loop(0, num_train, body_fn, (0.0, svi_state)) #fori_loop(lower, upper, body_fun, init_val)

@jit
def eval_test(rng_key, svi_state, num_test):

    def body_fn(i, loss_sum):
        rng_key_i = random.fold_in(rng_key, i) 
        rng_key_i, rng_key_ls, rng_key_var, rng_key_noise = random.split(rng_key_i, 4)
        batch = gp_predictive(rng_key_i, gp_kernel=args["gp_kernel"], x=args["x"], jitter=1e-4)
        loss = svi.evaluate(svi_state, batch['y']) / args['batch_size']
        loss_sum += loss
        return loss_sum

    loss = lax.fori_loop(0, num_test, body_fn, 0.0)
    loss = loss / num_test

    return loss

args = {"num_epochs": uargs.num_epochs, 
        "learning_rate": 1.0e-3, 
        "batch_size": 100, 
        "hidden_dim1": uargs.hidden_dim1,
        "hidden_dim2": uargs.hidden_dim2,
        "z_dim": uargs.z_dim,
         "x": x_xy,
        "n": len(x_xy),
        "gp_kernel": exp_sq_kernel,
        "rng_key": random.PRNGKey(1),
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "thinning": 1
        }

adam = optim.Adam(step_size=args["learning_rate"])

svi = SVI(vae_model, vae_guide, adam, Trace_ELBO(), 
          hidden_dim1=args["hidden_dim1"], 
          hidden_dim2=args["hidden_dim2"], 
          z_dim=args["z_dim"])

rng_key, rng_key_samp, rng_key_init = random.split(args["rng_key"], 3)
init_batch = gp_predictive(rng_key_predict, x=args["x"], gp_kernel = args["gp_kernel"])['y']
svi_state = svi.init(rng_key_init, init_batch)

test_loss_list = []

for i in range(args['num_epochs']):
    
    rng_key, rng_key_train, rng_key_test, rng_key_infer = random.split(rng_key, 4)
    
    t_start = time.time()

    num_train = 1000
    _, svi_state = epoch_train(rng_key_train, svi_state, num_train)

    num_test = 1000
    test_loss = eval_test(rng_key_test, svi_state, num_test)
    test_loss_list += [test_loss]

    print(
        "Epoch {}: loss = {} ({:.2f} s.)".format(
            i, test_loss, time.time() - t_start
        )
    )
    params = svi.get_params(svi_state)
    with open(f'output/decoder/svi_state_{i}.pkl','wb') as f:
        pickle.dump(params,f)
        
    if math.isnan(test_loss): break

decoder_params = svi.get_params(svi_state)["decoder$params"]
args["decoder_params"] = decoder_params
with open('output/decoder/2d_decoder.pkl','wb') as f:
    pickle.dump(decoder_params,f)
