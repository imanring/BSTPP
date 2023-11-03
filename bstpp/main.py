# general libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, ceil
import warnings 
import dill
import pickle

# JAX
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist

from bstpp.utils import * 
from inference_functions import *



class Point_Process_Model:
    def __init__(self,data,A,model='cox_hawkes',spatial_cov=None,interpolation=False,**priors):
        """
        Initialize Model
        Parameters:
            data (str or pd.DataFrame): either file path or DataFrame containing spatiotemporal data
                columns must include 'X', 'Y', 'T'
            A (tuple ((float,float),(float,float)) ): spatial region of interest. 
                First tuple is the x-range, second tuple is y-range
            model (str): one of ['cox_hawkes','lgcp','hawkes']
            spatial_cov (str,pd.DataFrame): either file path or DataFrame containing spatial covariates
                if interpolation is false there must be exactly 625 rows corresponding to the spatial grid cells
            interpolation (bool): interpolate covariates to center of covariate grid cells. **Not Implemented**
            priors (key word arguments): priors for parameters (a_0,w,alpha,beta,sigmax_2). Must be a numpyro distribution
        """
        if type(data)==str:
            df = pd.read_csv(data)
        elif type(data)==pd.DataFrame:
            df = data
        self.df = df
        
        args={}
        args['T']=50
        # Spatial grid is 1x1
        args['t_min']=0
        args['x_min']=0
        args['x_max']=1
        args['y_min']=0
        args['y_max']=1
        args['model']=model
        
        t_events_total=((df['T']-df['T'].min())).to_numpy()
        t_events_total/=t_events_total.max()
        t_events_total*=50

        x_range = A[0]
        x_events_total=(df['X']-x_range[0]).to_numpy()
        x_events_total/=(x_range[1]-x_range[0])

        y_range = A[1]
        y_events_total=(df['Y']-y_range[0]).to_numpy()
        y_events_total/=(y_range[1]-y_range[0])
        
        xy_events_total=np.array((x_events_total,y_events_total)).transpose()
        
        
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
        
        n=n_xy
        if args['model'] in ['lgcp','cox_hawkes']:
            with open('decoders/decoder_1d_T50_fixed_ls', 'rb') as file:
                decoder_params = pickle.load(file)
            args["decoder_params_temporal"] = decoder_params#Load 2d spatial trained decoder
            
            with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n_xy), 'rb') as file:
                decoder_params = pickle.load(file)
            args["decoder_params_spatial"] = decoder_params
        
        args["t_events"]=t_events_total
        args["xy_events"]=xy_events_total.transpose()

        if spatial_cov is not None:
            if type(spatial_cov)==str:
                spatial_cov = pd.read_csv(spatial_cov)
            # standardize covariates
            args['spatial_cov'] = (spatial_cov.values-spatial_cov.values.mean(axis=0))/(spatial_cov.values.var(axis=0)**0.5)
            args['num_cov'] = len(spatial_cov.columns)
            self.cov_names = list(spatial_cov.columns)
            
        if False:
            args['spatial_grid_cells'] = np.setdiff1d(np.arange(25**2),
                                                      array([0,1,2,3,4,10,25,26,27,28,50,51,52,53,75,76,77]))
        else:
            args['spatial_grid_cells'] = np.arange(25**2)

        default_priors = {"a_0": dist.Normal(2,2) if model=='lgcp' else dist.Normal(0,3),
                          "w": dist.Normal(jnp.zeros(args['num_cov']),jnp.ones(args["num_cov"])),
                          "alpha": dist.HalfNormal(0.5),
                          "beta": dist.HalfNormal(0.3),
                          "sigmax_2": dist.HalfNormal(1),
                         }
        
        for par, prior in kwargs:
            if par in default_priors:
                default_priors[par] = prior
            else:
                warnings.warn(f'Warning: {par} prior is not being used. There is no such parameter.') 
        args['priors'] = default_priors
        
        self.args = args
    
    def run_mcmc(self,batch_size=1,num_warmup=500,num_samples=1000,num_chains=1,thinning=1):
        """
        Run MCMC posterior sampling on model
        Parameters:
            See numpyro documentation for description
        """
        self.args["batch_size"]= batch_size
        self.args["num_warmup"]= num_warmup
        self.args["num_samples"] = num_samples
        self.args["num_chains"] = num_chains
        self.args["thinning"] = thinning
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        
        output_dict = {}
        if self.args['model'] in ['cox_hawkes','hawkes']:
            output_dict['model']=spatiotemporal_hawkes_model
            self.mcmc = run_mcmc(rng_key_post, spatiotemporal_hawkes_model, self.args)
        elif self.args['model'] == 'lgcp':
            output_dict['model']=spatiotemporal_LGCP_model
            self.mcmc = run_mcmc(rng_key_post, spatiotemporal_LGCP_model, self.args)
        self.mcmc_samples=self.mcmc.get_samples()
        
        output_dict['samples']=self.mcmc_samples
        output_dict['mcmc']=self.mcmc
        with open('output/'+self.args['model']+'/output.pkl', 'wb') as handle:
            dill.dump(output_dict, handle)

    def plot_trigger_posterior(self,output_file=None):
        """
        Plot histograms of posterior trigger parameters
        Parameters:
            output_file (str): path in which to save plot
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['hawkes','cox_hawkes']:
            raise Exception("This is not a Hawkes Model. Cannot plot trigger parameters.")
        fig, ax = plt.subplots(1, 3,figsize=(8,4), sharex=False)
        plt.suptitle("Trigger Parameter Posteriors")
        ax[0].hist(self.mcmc_samples['alpha'])
        ax[0].set_xlabel(r"${\alpha} $")
        ax[1].hist(self.mcmc_samples['beta'])
        ax[1].set_xlabel(r"$\beta$")
        ax[2].hist(self.mcmc_samples['sigmax_2']**0.5)
        ax[2].set_xlabel('$\sigma$')
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()

    def plot_trigger_time_decay(self,output_file=None,t_units='days'):
        """
        Plot temporal trigger kernel sample posterior
        Parameters:
            output_file (str): path in which to save plot
            t_units (str): time units of original data
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['hawkes','cox_hawkes']:
            raise Exception("This is not a Hawkes Model. Cannot plot trigger parameters.")
        
        scale = 50/self.df['T'].max()
        post_mean = float(self.mcmc_samples['beta'].mean())
        x = np.arange(0,3.5/(scale*post_mean),post_mean*3.5/250)
        fig, ax = plt.subplots(figsize=(7,7))
        for b in np.random.choice(self.mcmc_samples['beta'],100):
            plt.plot(x,b*np.exp(-b*scale*x),color='black',alpha=0.1)
        fig.suptitle('Time Decay of Trigger Function')
        ax.set_ylabel('Trigger Intensity')
        ax.set_xlabel(t_units.capitalize()+' After First Event')
        ax.axhline(0,color='black',linestyle='--')
        ax.axvline(0,color='black',linestyle='--')
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
        print(f'Mean trigger time: {round(1/(post_mean*scale),2)} '+t_units)

    def cov_weight_post_summary(self,plot_file=None,summary_file=None):
        """
        Plot posteriors of weights and bias and save summary of posteriors]
        Parameters:
            plot_file (str): path in which to save plot
            summary_file (str): path in which to save summary
        Returns:
            (pd.DataFrame): summary of weights and bias
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if 'spatial_cov' not in self.args:
            raise Exception("Spatial covariates were not included in the model.")
        
        n = self.mcmc_samples['w'].shape[1]+1
        c = ceil(n**0.5)
        r = ceil(n/c)
        fig, ax = plt.subplots(r,c,figsize=(10,10), sharex=False)
        fig.suptitle('Covariate Weights', fontsize=16)
        for i in range(n-1):
            ax[i//c,i%c].hist(self.mcmc_samples['w'].T[i])
            ax[i//c,i%c].set_xlabel(self.cov_names[i])
        ax[(n-1)//c,(n-1)%c].hist(self.mcmc_samples['a_0'])
        ax[(n-1)//c,(n-1)%c].set_xlabel("$a_0$")
        if plot_file is not None:
            plt.savefig(plot_file)
        plt.show()

        
        w_samps = np.concatenate((self.mcmc_samples['w'],self.mcmc_samples['a_0'].reshape(-1,1)),axis=1)
        mean = w_samps.mean(axis=0)
        std = w_samps.var(axis=0)**0.5
        z_score = np.asarray(mean/std)
        p_val = 1-np.vectorize(erf)(abs(z_score)/2**0.5)
        quantiles = np.quantile(w_samps,[0.025,0.975],axis=0)
        w_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'z':z_score,'P>|z|':p_val,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=self.cov_names+['a_0'])
        if summary_file is not None:
            w_summary.to_csv(summary_file)
        return w_summary
    
    def plot_temporal_background(self,output_file=None):
        """
        Plot mean posterior temporal gaussian process
        Parameters:
            output_file (str): path in which to save plot
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: temporal background in constant.")
        
        x_t = jnp.arange(0, self.args['T'], self.args['T']/self.args["n_t"])
        f_t_post=self.mcmc_samples["f_t"]
        f_t_post_mean=jnp.mean(f_t_post, axis=0)
        
        fig,ax=plt.subplots(1,1,figsize=(8,5))
        event_time_height = np.ones(len(self.args['t_events']))*(f_t_post_mean.min()-f_t_post_mean.var()**0.5/4)
        ax.plot(self.args['t_events'], event_time_height,'+',color="red", label="observed times")
        ax.set_ylabel('$f_t$')
        ax.set_xlabel('t')
        ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
        ax.legend()
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
        
    def plot_spatial_background(self,output_file=None,include_cov=False):
        """
        Plot mean posterior spatial background with/without covariates
        Parameters:
            output_file (str): path in which to save plot
            include_cov (bool): include effects of spatial covariates
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp'] and not include_cov:
            raise Exception("Nothing to plot: spatial background is constant")
        if include_cov and 'spatial_cov' not in self.args:
            raise Exception("No spatial covariates are in the model and include_cov was set to True")
        
        if self.args['model'] in ['cox_hawkes','lgcp']:
            f_xy_post=self.mcmc_samples["f_xy"]
        else:
            f_xy_post = 0
        
        if include_cov:
            f_xy_post = f_xy_post + self.mcmc_samples['b_0']
        
        if self.args['model'] in ['cox_hawkes','lgcp'] and include_cov:
            fig_desc = "$f_s$ + X(s)w"
        elif include_cov:
            fig_desc = "X(s)w"
        else:
            fig_desc = "$f_s$"

        f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
        
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        _min, _max = np.amin(f_xy_post), np.amax(f_xy_post)
        im = ax[0].imshow(f_xy_post_mean.reshape(self.args["n_xy"], self.args["n_xy"]), 
                          cmap='viridis', interpolation='none', extent=[0,1,0,1], 
                          origin='lower',vmin=_min, vmax=_max)
        ax[0].title.set_text('Mean Posterior '+fig_desc)
        im2 = ax[1].imshow(f_xy_post_mean.reshape(self.args["n_xy"], self.args["n_xy"]), 
                           cmap='viridis', interpolation='none', extent=[0,1,0,1], 
                           origin='lower',vmin=_min, vmax=_max)
        ax[1].plot(self.args["xy_events"][0],self.args["xy_events"][1],'x', 
                   alpha=.25,color='red',label='true event locations')
        ax[1].title.set_text(f'Mean Posterior {fig_desc} With Events')
        for i in range(2):
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            for item in ([ ax[i].xaxis.label, ax[i].yaxis.label, ax[i].title] ):
                item.set_fontsize(15)
        
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([ax[1].get_position().x1+0.03,ax[1].get_position().y0,0.02,ax[1].get_position().height])
        fig.colorbar(im, cax=cax)
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()