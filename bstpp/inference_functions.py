import os
import time 
import jax
import jax.numpy as jnp

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import Trace_ELBO, MCMC, NUTS, init_to_median
from .utils import difference_matrix, difference_matrix_partial
from .vae_functions import *


def spatiotemporal_hawkes_model(args):
    t_events=args["t_events"]
    xy_events=args["xy_events"]
    N=t_events.shape[0]

    if args['model'] == 'hawkes':
      a_0 = numpyro.sample("a_0", args['priors']['a_0'])
      if 'spatial_cov' in args:
        w = numpyro.sample("w", args['priors']['w'])
        b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)
      else:
        b_0=0
      mu_xyt=numpyro.deterministic("mu_xyt",jnp.exp(a_0+b_0))
      if 'spatial_cov' in args:
        Itot_txy_back = numpyro.deterministic("Itot_txy_back",mu_xyt@args['cov_area']*args['T'])
        mu_xyt_events = mu_xyt[args["cov_ind"]]
      else:
        Itot_txy_back = numpyro.deterministic("Itot_txy_back",mu_xyt*args['T']*args['A_area'])
        mu_xyt_events = mu_xyt

    ####### LGCP BACKGROUND
    if args['model']=='cox_hawkes':
      # Intercept of linear combination
      a_0 = numpyro.sample("a_0", args['priors']['a_0'])
      
      # Generate gaussian vector to feed into VAE
      z_temporal = numpyro.sample("z_temporal", 
                                  dist.Normal(jnp.zeros(args["z_dim_temporal"]), 
                                              jnp.ones(args["z_dim_temporal"]))
                                 )
      decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
      decoder_params = args["decoder_params_temporal"]
      # Approximate Gaussian Process with VAE
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
      rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
      # calculate temporal integral over LGCP
      Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])
      # Temporal part of log(mu(t,s))
      f_t_events=f_t[args["indices_t"]]

      # Generate gaussian vector to feed into VAE
      z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
      decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
      decoder_params = args["decoder_params_spatial"]
      # Generate Gaussian Process from VAE
      f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
      f_xy_events=f_xy[args["indices_xy"]]
      
      # Calculate spatial intensity
      if 'spatial_cov' in args:
          # weights for linear combination
          w = numpyro.sample("w", args['priors']['w'])
          b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)
          
          f_xy_events = f_xy_events + b_0[args['cov_ind']]
          spatial_integral = jnp.exp(b_0[args['int_df']['cov_ind'].values] + 
                                     f_xy[args['int_df']['comp_grid_id'].values]) @ args['int_df']['area'].values
      else:
          rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy))
          spatial_integral = jnp.mean(rate_xy[args['spatial_grid_cells']])
      Itot_xy=numpyro.deterministic("Itot_xy", spatial_integral)
      
      #Calculate total background integral
      Itot_txy_back=numpyro.deterministic("Itot_txy_back",Itot_t*Itot_xy)


    #### EXPONENTIAL KERNEL for the excitation part
    #temporal exponential kernel parameters
    alpha = numpyro.sample("alpha", args['priors']['alpha'])
    beta = numpyro.sample("beta", args['priors']['beta'])
    
    #spatial gaussian kernel parameters     
    sigmax_2 = numpyro.sample("sigmax_2", args['priors']['sigmax_2'])
    sigmay_2 = sigmax_2
    
    
    T,x_min,x_max,y_min,y_max = args['T'],args['x_min'],args['x_max'],args['y_min'],args['y_max']  
    
    T_diff=difference_matrix(t_events);
    S_mat_x = difference_matrix(xy_events[0])
    S_mat_y = difference_matrix(xy_events[1])
    S_diff_sq=(S_mat_x**2)/sigmax_2+(S_mat_y**2)/sigmay_2; 
    l_hawkes_sum=alpha*beta/(2*jnp.pi*jnp.sqrt(sigmax_2*sigmay_2))*jnp.exp(-beta*T_diff-0.5*S_diff_sq)
    l_hawkes = numpyro.deterministic('l_hawkes',jnp.sum(jnp.tril(l_hawkes_sum,-1),1))

    if args['model'] == 'hawkes':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+mu_xyt_events)))
    elif args['model']=='cox_hawkes':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+jnp.exp(a_0 + f_t_events+f_xy_events))))

    #### hawkes integral
    exponpart = alpha*(1-jnp.exp(-beta*(T-t_events)))
    numpyro.deterministic("exponpart",exponpart)
    
    s1max=(x_max-xy_events[0])/(jnp.sqrt(2*sigmax_2))
    s1min=(xy_events[0])/(jnp.sqrt(2*sigmax_2))
    gaussianpart1=0.5*jax.scipy.special.erf(s1max)+0.5*jax.scipy.special.erf(s1min)
    
    s2max=(y_max-xy_events[1])/(jnp.sqrt(2*sigmay_2))
    s2min=(xy_events[1])/(jnp.sqrt(2*sigmay_2))
    gaussianpart2=0.5*jax.scipy.special.erf(s2max)+0.5*jax.scipy.special.erf(s2min)
    gaussianpart=gaussianpart2*gaussianpart1
    numpyro.deterministic("gaussianpart",gaussianpart)    

    ## total integral
    Itot_txy=jnp.sum(exponpart*gaussianpart)+Itot_txy_back
    numpyro.deterministic("Itot_txy",Itot_txy)
    loglik=numpyro.deterministic('loglik',ell_1-Itot_txy)

    numpyro.factor("t_events", loglik) 
    numpyro.factor("xy_events", loglik)


def spatiotemporal_LGCP_model(args):
    t_events=args["t_events"];
    xy_events=args["xy_events"];
    n_obs=t_events.shape[0]
    
    #temporal rate
    a_0 = numpyro.sample("a_0", args['priors']['a_0'])
    
    #zero mean temporal gp 
    z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
    decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
    decoder_params = args["decoder_params_temporal"]
    v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
    f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
    rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
    Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])
    f_t_i=f_t[args["indices_t"]]
    
    # zero mean spatial gp
    z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
    decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
    decoder_params = args["decoder_params_spatial"]
    f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
    rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy))
    f_xy_i=f_xy[args["indices_xy"]]
    
    if 'spatial_cov' in args:
        # weights for linear combination
        w = numpyro.sample("w", args['priors']['w'])
        b_0 = numpyro.deterministic("b_0", args['spatial_cov'] @ w)
        f_xy_i += b_0[args['cov_ind']]
        spatial_integral = jnp.exp(b_0[args['int_df']['cov_ind'].values] +
                                   f_xy[args['int_df']['comp_grid_id'].values]
                                  ) @ args['int_df']['area'].values
    else:
        spatial_integral = jnp.mean(rate_xy[args['spatial_grid_cells']])
    
    Itot_xy=numpyro.deterministic("Itot_xy", spatial_integral)
    

    loglik=jnp.sum(f_t_i+f_xy_i+a_0)
    I_tot_txy=numpyro.deterministic("I_tot_txy",Itot_xy*Itot_t)
    loglik-=I_tot_txy
    numpyro.deterministic("loglik",loglik)

    numpyro.factor("t_events", loglik)
    numpyro.factor("xy_events", loglik)


def run_mcmc(rng_key, model_mcmc, args):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model_mcmc, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, args)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc