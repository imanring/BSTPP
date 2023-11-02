# general libraries
import os
import numpy as np
import pickle

# JAX
import jax.numpy as jnp
from jax import random

import pandas as pd
from utils import * 
from inference_functions import *
import dill


df = pd.read_csv (r'data/pp_data.csv')
df['X'] = df.X + np.random.normal(scale=4/69,size=len(df.X))
df['Y'] = df.Y + np.random.normal(scale=4/69,size=len(df.Y))

# arguments for model
args={}
# Temporal grid is 50 units long
args['T']=50
# Spatial grid is 1x1
args['t_min']=0
args['x_min']=0
args['x_max']=1
args['y_min']=0
args['y_max']=1
args['background']='LGCP'


t_events_total=((df['T']-df['T'].min())).to_numpy()
t_events_total/=t_events_total.max()
t_events_total*=50
t_events_total

x_range = (3,15.5)
x_events_total=(df['X']-x_range[0]).to_numpy()
x_events_total/=(x_range[1]-x_range[0])
x_events_total

y_range = (4,16.5)
y_events_total=(df['Y']-y_range[0]).to_numpy()
y_events_total/=(y_range[1]-y_range[0])
y_events_total

xy_events_total=np.array((x_events_total,y_events_total)).transpose()


if args['background']=='LGCP':
  rng_key, rng_key_predict = random.split(random.PRNGKey(10))

  n_t=50
  T=50
  x_t = jnp.arange(0, T, T/n_t)
  args[ "n_t"]=n_t
  args["x_t"]=x_t

  n_xy = 25
  grid = jnp.arange(0, 1, 1/n_xy)
  u, v = jnp.meshgrid(grid, grid)
  x_xy = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
  args['x_xy']=x_xy
  args["n_xy"]= n_xy
  
  args["gp_kernel"]=exp_sq_kernel
  args["batch_size"]= 1

  indices_t=find_index(t_events_total, x_t)
  indices_xy=find_index(xy_events_total, x_xy)
  args['indices_t']=indices_t
  args['indices_xy']=indices_xy



# temporal VAE training arguments
args["hidden_dim_temporal"]= 35
args["z_dim_temporal"]= 11
args["T"]=T
# spatial VAE training arguments
args["hidden_dim1_spatial"]= 35
args["hidden_dim2_spatial"]= 30
args["z_dim_spatial"]=10
n_xy=25


## I load my 1D temporal trained decoder parameters to generate GPs with hyperparameters that make sense in this domain
# Load 
# fixed lengthscale=10, var=1, T50
with open('decoders/decoder_1d_T50_fixed_ls', 'rb') as file:
    decoder_params = pickle.load(file)
    print(len(decoder_params))

args["decoder_params_temporal"] = decoder_params

n=n_xy
if args['background']=='LGCP':
  #Load 2d spatial trained decoder
  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n_xy), 'rb') as file:
      decoder_params = pickle.load(file)
      print(len(decoder_params))

  args["decoder_params_spatial"] = decoder_params


t_events=t_events_total;
xy_events=xy_events_total


# MCMC inference
args["num_warmup"]= 500
args["num_samples"] = 1000
args["num_chains"] =1
args["thinning"] =1


args["t_events"]=t_events_total
args["xy_events"]=xy_events_total.transpose()

spatial_cov = pd.read_csv('data/spatial_cov.csv')
columns = ['droughtstart_speibase','urban_ih_log','droughtyr_speigdm','herb_gc','capdist',
           'grass_ih_log','nlights_sd_log','water_gc_log','pop_gpw_sd_log','pasture_ih']
args['spatial_cov'] = spatial_cov[columns].values
# standardize covariates
args['spatial_cov'] = (args['spatial_cov']-args['spatial_cov'].mean(axis=0))/(args['spatial_cov'].var(axis=0)**0.5)
args['num_cov'] = len(columns)

if False:
    args['spatial_grid_cells'] = np.setdiff1d(np.arange(25**2),
                                              array([0,1,2,3,4,10,25,26,27,28,50,51,52,53,75,76,77]))
else:
    args['spatial_grid_cells'] = np.arange(25**2)


rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)

# inference
mcmc = run_mcmc(rng_key_post, spatiotemporal_hawkes_model, args)
mcmc_samples=mcmc.get_samples()


import dill
output_dict = {}
output_dict['model']=spatiotemporal_hawkes_model
output_dict['samples']=mcmc.get_samples()
output_dict['mcmc']=mcmc
with open('output.pkl', 'wb') as handle:
    dill.dump(output_dict, handle)

