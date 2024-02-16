# general libraries
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops
#from jax.experimental import stax

from functools import partial


#@title

def square_mean(a,b):
  return np.mean((a-b)**2)

def sq_diff(a,b):
  return np.sum((a-b)**2)
  
## helper functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx.astype(int)

def find_nearest_2D(array, value):
    array = np.asarray(array)
    idx = (np.sqrt(np.sum((array-value)**2,1))).argmin()
    return idx

    
def randdist(x, pdf, nvals):
    """Produce nvals random samples from pdf(x), assuming constant spacing in x."""
    # get cumulative distribution from 0 to 1
    cumpdf = np.cumsum(pdf)
    cumpdf *= 1/cumpdf[-1]

    # input random values
    randv = np.random.uniform(size=nvals)

    # find where random values would go
    idx1 = np.searchsorted(cumpdf, randv)
    # get previous value, avoiding division by zero below
    idx0 = np.where(idx1==0, 0, idx1-1)
    idx1.at[idx0==0].set(1)

    # do linear interpolation in x
    frac1 = (randv - cumpdf[idx0]) / (cumpdf[idx1] - cumpdf[idx0])
    if x.size==x.shape[0]:
        randdist = x[idx0]*(1-frac1) + x[idx1]*frac1
    elif x.size==x.shape[0]*2:
        randdist = x[idx0,:]*(1-frac1).reshape(idx0.size,1) + x[idx1,:]*frac1.reshape(idx0.size,1) 
    randdist_y = pdf[idx0]*(1-frac1) + pdf[idx1]*frac1
    indices = randdist>0
    return idx0, idx1, randdist[indices], randdist_y[indices]



    #@title
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


def exp_sq_kernel(x, z, var, length, noise, jitter=1.0e-6):
    dist = dist_euclid(x, z)
    deltaXsq = jnp.power(dist/ length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    k += (noise + jitter) * jnp.eye(x.shape[0])
    return k

    
def GP(args, jitter=1e-4, y=None, var=None, length=None, sigma=None, noise=False):
    
    x = args["x"]
    #obs_idx = args["obs_idx"]
    gp_kernel=args["gp_kernel"] 

    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4,1))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
    
    k = gp_kernel(x, x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.05))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)



#@title
def rej_sampling_new(N, grid, gp_function, n):
  f_max=np.max(gp_function);
  ids=np.arange(0, n)
  if N<100:
    N_max=N*100
  else:
    N_max=N*10;
  index=np.random.choice(ids,N_max)

  candidate_points=grid[index];

  #plt.plot(args['x_t'],gp_function)
  #plt.plot(args['x_t'][index],gp_function[index],'x')
  U=np.random.uniform(0, f_max, N_max);
  #plt.plot(x_t[index],U,'o')
  indices=jnp.where(U<gp_function[index]);
  #plt.plot(x_t[index][indices],U[indices],'+' )
  #plt.plot(x_t[index][indices],np.zeros(indices[0].size),'+' )
  accepted_points=grid[index][indices][0:N]
  accepted_f=gp_function[index][indices][0:N]
  return jnp.array(index[indices][0:N]), accepted_points, accepted_f



def find_index_b(events, grid):
  n=events.shape[0];
  ind=np.zeros(n)*np.nan
  for i in range(n):  
    if events.size>events.shape[0]:
      ind[i]=jnp.nanargmin(np.sqrt(np.sum((events[i,:]-grid)**2,1)))
    else:
      ind[i]=jnp.nanargmin(np.abs(events[i]-grid))
  return ind.astype(int)
 

#@title
def find_index(events, grid):
  if isinstance(events,np.ndarray):
    n=events.shape[0];
    ind=np.zeros(n)*np.nan
    for i in range(n):  
      if events.size>events.shape[0]:
        ind[i]=jnp.nanargmin(np.sqrt(np.sum((events[i,:]-grid)**2,1)))
      else:
        ind[i]=jnp.nanargmin(np.abs(events[i]-grid))
  else:
    ind=jnp.nanargmin(np.abs(events-grid))
  return ind.astype(int)
  

  #@title

def difference_matrix(a):
    x = jnp.reshape(a, (len(a), 1))
    x2= x- x.transpose()
    x2trial=jnp.tril(x2)
    return x2trial

def difference_matrix_partial(a,partial_index):
    x = jnp.reshape(a, (len(a), 1))
    x2= x[partial_index]- x.transpose()
    x2trial=jnp.tril(x2)
    return x2trial

def sq_diff(a,b):
  return np.sum((a-b)**2)


def square_mean(a,b):
  return np.mean((a-b)**2)
